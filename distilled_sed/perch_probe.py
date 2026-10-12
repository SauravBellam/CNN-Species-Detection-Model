"""Ablation: frozen Perch v2 + attention pooling + linear probe (no CNN).

Uses the per-frame Perch embeddings cached by pseudo_label.py
(*_spatial.npy, (T, 1536) per 5 s window; falls back to the single 1536-d
embedding if Perch exposed no spatial output). Per class, a linear layer
scores every frame and a second linear layer gives attention weights over the
frames -- the same attention-pooling head as the SED model, without the hidden layer.

Same training data, targets and folds as model1_sed.py (focal windows minus
the fold's clips, labelled soundscapes of other folds, pseudo-labelled
soundscapes), validated on the same soundscape windows, written as oof.npz in
the same format so ablations/evaluate.py scores it like any other variant.
Perch itself is not run here: its embeddings were computed on the GPU by
pseudo_label.py. It answers: how good is the teacher alone, and what does running Perch on
site cost compared to the distilled ResNet?

    python -m distilled_sed.perch_probe --comp-dir $COMP_DIR \
        --waveform-cache $WAVEFORM_CACHE_DIR --pseudo-dir $PSEUDO_DIR \
        --out runs/ablations/perch_attn_probe
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold, StratifiedKFold

from .tracking import class_groups, eval_metrics, to_long, write_curves

N_FOLDS, SEED = 5, 42


class AttnPoolProbe(nn.Module):
    def __init__(self, dim, n_classes):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.att = nn.Linear(dim, n_classes)
        self.cla = nn.Linear(dim, n_classes)

    def forward(self, x):                     # x (B, T, D)
        x = self.norm(x)
        w = torch.softmax(torch.tanh(self.att(x)), dim=1)
        frame = self.cla(x)                   # (B, T, K)
        return (w * frame).sum(1), frame


def load_feats(pseudo_dir, prefix):
    p = pseudo_dir / f"{prefix}_spatial.npy"
    if p.exists():
        return np.load(p, mmap_mode="r")
    return np.load(pseudo_dir / f"{prefix}_emb.npy", mmap_mode="r")[:, None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp-dir", type=Path, required=True)
    ap.add_argument("--waveform-cache", type=Path, required=True)
    ap.add_argument("--pseudo-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(N_FOLDS)))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED); np.random.seed(SEED)

    primary = pd.read_csv(args.comp_dir / "sample_submission.csv").columns[1:].tolist()
    l2i = {l: i for i, l in enumerate(primary)}
    K = len(primary)

    # ---- folds, replicated from model1_sed.py
    acm = pd.read_csv(args.waveform_cache / "audio_cache_meta.csv")
    acm = acm[acm["primary_label"].isin(l2i)].drop_duplicates("original_idx").reset_index(drop=True)
    acm["fold"] = -1
    for f, (_, vi) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                                .split(acm, acm["primary_label"])):
        acm.loc[vi, "fold"] = f
    focal = pd.read_parquet(args.pseudo_dir / "focal_windows.parquet")
    focal["row"] = np.arange(len(focal))
    focal = focal.merge(acm[["original_idx", "fold"]], on="original_idx", how="inner")

    sc_meta = pd.read_csv(args.waveform_cache / "soundscape_cache_meta.csv")
    files = sc_meta[["filename", "site"]].drop_duplicates().reset_index(drop=True)
    files["fold"] = -1
    for f, (_, vi) in enumerate(GroupKFold(N_FOLDS).split(files, groups=files["filename"])):
        files.loc[files.index[vi], "fold"] = f
    sc_meta["fold"] = sc_meta["filename"].map(dict(zip(files["filename"], files["fold"]))).fillna(-1).astype(int)

    # ---- labels for the validation windows (same construction as model1_sed.py)
    raw = pd.read_csv(args.comp_dir / "train_soundscapes_labels.csv").drop_duplicates()
    raw["start_sec"] = pd.to_timedelta(raw["start"]).dt.total_seconds().astype(int)
    Y = np.zeros((len(sc_meta), K), np.float32)
    lab = raw.groupby(["filename", "start_sec"])["primary_label"].apply(lambda s: ";".join(s))
    for i, (fn, st) in enumerate(zip(sc_meta["filename"], sc_meta["start_sec"])):
        for l in str(lab.get((fn, int(st)), "")).split(";"):
            if l.strip() in l2i:
                Y[i, l2i[l.strip()]] = 1.0

    sc_win = pd.read_parquet(args.pseudo_dir / "sc_windows.parquet")
    sc_row = {(f, int(t)): i for i, (f, t) in enumerate(zip(sc_win["filename"], sc_win["start_sec"]))}
    sc_meta["row"] = [sc_row.get((f, int(t)), -1) for f, t in zip(sc_meta["filename"], sc_meta["start_sec"])]

    FX, FT = load_feats(args.pseudo_dir, "focal"), np.load(args.pseudo_dir / "focal_targets.npy", mmap_mode="r")
    SX = load_feats(args.pseudo_dir, "sc")
    unl_rows = np.where(~sc_win["labeled"].values)[0]

    oof = {k: np.full((len(sc_meta), K), np.nan, np.float32) for k in ("clip", "fmax", "blend")}
    train_df = pd.read_csv(args.comp_dir / "train.csv")
    tax = pd.read_csv(args.comp_dir / "taxonomy.csv")
    groups = class_groups(primary, dict(zip(tax["primary_label"].astype(str), tax["class_name"].astype(str))),
                          train_df["primary_label"].astype(str).value_counts().reindex(primary).fillna(0).values)
    ns22_all = sc_meta["site"].values != "S22"
    rows, per_epoch = [], {}   # per_epoch[ep] = list of (val index, probs, train_loss)

    def predict(model, va):
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(SX[va["row"].values], np.float32)).to(dev)
            clip, frame = model(x)
            return torch.sigmoid(clip).cpu().numpy(), torch.sigmoid(frame.max(1).values).cpu().numpy()
    for k in args.folds:
        ST = np.load(args.pseudo_dir / f"sc_targets_fold{k}.npy", mmap_mode="r")
        tr_lab = sc_meta[(sc_meta["fold"] != k) & (sc_meta["row"] >= 0)]
        # (source array, row index, target)
        items = ([("f", r, None) for r in focal.loc[focal["fold"] != k, "row"]]
                 + [("s", r, Y[i]) for i, r in zip(tr_lab.index, tr_lab["row"])]
                 + [("s", r, None) for r in unl_rows])
        model = AttnPoolProbe(SX.shape[-1], K).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        def batch(sel):
            xs, ys = [], []
            for src, r, y in sel:
                xs.append(np.asarray((FX if src == "f" else SX)[r], np.float32))
                ys.append(y if y is not None else np.asarray((FT if src == "f" else ST)[r], np.float32))
            return torch.from_numpy(np.stack(xs)).to(dev), torch.from_numpy(np.stack(ys)).to(dev)

        va = sc_meta[(sc_meta["fold"] == k) & (sc_meta["row"] >= 0)]
        for ep in range(args.epochs):
            t0 = time.time()
            model.train()
            perm = np.random.permutation(len(items))
            tot = 0.0
            for s in range(0, len(perm), args.batch):
                x, y = batch([items[i] for i in perm[s:s + args.batch]])
                clip, frame = model(x)
                loss = 0.5 * F.binary_cross_entropy_with_logits(clip, y) \
                    + 0.5 * F.binary_cross_entropy_with_logits(frame.max(1).values, y)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item()
            tl = tot / max(1, -(-len(perm) // args.batch))
            pc, pf = predict(model, va)
            pb = 0.5 * pc + 0.5 * pf
            em = eval_metrics(Y[va.index], pb, va["filename"].values, va["start_sec"].values,
                              ns22_all[va.index], groups)
            em.update(train_loss=tl, epoch_seconds=time.time() - t0)
            rows += to_long(args.out.name, f"fold{k}", ep, em)
            per_epoch.setdefault(ep, []).append((va.index.values, pb, tl, em["epoch_seconds"]))
            print(f"fold {k} ep {ep:02d} loss {tl:.4f} val auc {em['auc_soundscape_macro']:.4f}", flush=True)

        oof["clip"][va.index], oof["fmax"][va.index] = pc, pf
        oof["blend"][va.index] = 0.5 * pc + 0.5 * pf

    for ep, parts in sorted(per_epoch.items()):
        idx = np.concatenate([a for a, _, _, _ in parts])
        em = eval_metrics(Y[idx], np.concatenate([b for _, b, _, _ in parts]),
                          sc_meta["filename"].values[idx], sc_meta["start_sec"].values[idx],
                          ns22_all[idx], groups)
        em["train_loss"] = float(np.mean([c for _, _, c, _ in parts]))
        em["epoch_seconds"] = float(np.mean([d for _, _, _, d in parts]))
        rows += to_long(args.out.name, "pooled", ep, em)
    write_curves(args.out / "curves.csv", rows)
    np.savez_compressed(
        args.out / "oof.npz", y=Y, ns22=sc_meta["site"].values != "S22",
        filename=sc_meta["filename"].values.astype(str), start_sec=sc_meta["start_sec"].values.astype(int),
        fold=sc_meta["fold"].values.astype(int),
        labels=np.array(primary), **oof,
        train_counts=train_df["primary_label"].astype(str).value_counts().reindex(primary).fillna(0).astype(int).values)
    print(f"Saved {args.out / 'oof.npz'}")


if __name__ == "__main__":
    main()
