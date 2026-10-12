"""Offline Perch v2 pseudo-labelling of 5 s windows (run once, before training).

The labelling system is taken from the Perch pipeline (Model_22 / "V17") in
nina2025's Kaggle notebook "BirdCLEF 2026 EoS.9"
(https://www.kaggle.com/code/nina2025/birdclef-2026-eos-9):
  * Perch label -> competition species mapping by scientific name;
  * genus proxies (max over Perch classes of the same genus) for unmapped
    non-sonotype species in Aves / Amphibia / Insecta;
  * fold-safe site / hour / site-hour prior tables fused in logit space
    (lambda_event 0.4, lambda_texture 1.0, lambda_proxy_texture 0.8);
  * texture (frogs/insects) neighbour smoothing 0.35, event (birds) local-max
    smoothing 0.15 across the 12 windows of a 60 s file; unmapped species never
    seen in labelled soundscapes are set to -8.

What it produces (all windows are exactly 5 s):
  focal_windows.parquet + focal_targets.npy + focal_emb.npy [+ focal_spatial.npy]
      Every focal train_audio clip cut into 5 s windows. The clip only says WHICH
      species are present; Perch says WHEN. target[w, c] = p_c(w) / max_w p_c(w)
      for the clip's primary + secondary labels, 0 for all other species. If
      Perch cannot see the species at all (max p < --min-detect) the clip label
      is kept as 1 for every window.
  sc_windows.parquet + sc_targets_fold{k}.npy + sc_emb.npy [+ sc_spatial.npy]
      All 12 windows of every train soundscape. Labelled files keep their ground
      truth; unlabelled files get the fused Perch + prior score above. Priors
      for fold k exclude fold-k files, so validation labels never leak in.
  sc_wave/<stem>.npy   int16 audio of unlabelled soundscapes, for training.
  *_emb.npy            Perch 1536-d embedding per window (distillation target).
  *_spatial.npy        Perch per-frame embeddings (T, 1536) per window, for the
                       Perch + attention-pooling + linear-probe ablation.

Perch runs on the GPU (onnxruntime-gpu); the script stops if CUDA is not available.
Usage (on the VM, onnxruntime-gpu installed):
  python -m distilled_sed.pseudo_label \
      --comp-dir $COMP_DIR --waveform-cache $WAVEFORM_CACHE_DIR \
      --perch-onnx /data/perch_v2.onnx --perch-labels /data/perch_labels.csv \
      --out $PSEUDO_DIR
--perch-onnx must expose "label" and "embedding" outputs (e.g. rishikeshjani's
perch-onnx-for-birdclef-2026). --perch-labels is Perch v2's assets/labels.csv.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.model_selection import GroupKFold

from .priors import fit_prior_tables, prior_logits, site_hour

SR = 32000
WIN = 5 * SR
N_WINDOWS = 12
N_FOLDS = 5
BEST = {"lambda_event": 0.4, "lambda_texture": 1.0, "lambda_proxy_texture": 0.8,
        "smooth_texture": 0.35, "smooth_event": 0.15}
TEXTURE_TAXA = {"Amphibia", "Insecta"}
PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}


# ---------------------------------------------------------------- Perch
class Perch:
    """Perch v2 as ONNX on the GPU (onnxruntime-gpu, CUDAExecutionProvider).

    Not the "perch_v2_cpu" TensorFlow SavedModel the Kaggle notebook used: that
    variant is built for CPU-only scoring. The ONNX graph is device-agnostic and
    runs on CUDA when onnxruntime-gpu is installed. We refuse to silently fall
    back to CPU (it would take many times longer) unless allow_cpu is set.
    """

    def __init__(self, onnx_path, batch=64, allow_cpu=False):
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers() and not allow_cpu:
            raise RuntimeError("onnxruntime has no CUDA provider -- install onnxruntime-gpu "
                               "matching the VM's CUDA (or pass --allow-cpu)")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            str(onnx_path), so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        print("Perch providers:", self.sess.get_providers())
        if self.sess.get_providers()[0] != "CUDAExecutionProvider" and not allow_cpu:
            raise RuntimeError("Perch session did not start on CUDA (check CUDA/cuDNN libs)")
        self.inp = self.sess.get_inputs()[0].name
        self.out = {o.name: i for i, o in enumerate(self.sess.get_outputs())}
        assert "label" in self.out and "embedding" in self.out, f"outputs: {list(self.out)}"
        self.spatial_key = next((k for k in self.out if "spatial" in k), None)
        self.batch = batch

    def __call__(self, x):
        """x (N, 160000) float32 -> logits (N, L), emb (N, 1536), spatial (N, T, 1536) or None."""
        L, E, S = [], [], []
        for s in range(0, len(x), self.batch):
            outs = self.sess.run(None, {self.inp: x[s:s + self.batch]})
            L.append(outs[self.out["label"]]); E.append(outs[self.out["embedding"]])
            if self.spatial_key:
                sp = outs[self.out[self.spatial_key]]
                S.append(sp.mean(axis=2) if sp.ndim == 4 else sp)  # (N, T, F, D) -> (N, T, D)
        cat = lambda a: np.concatenate(a).astype(np.float32)
        return cat(L), cat(E), (cat(S) if S else None)


# ------------------------------------------------- mapping (Model_22 cell 3)
def build_mapping(taxonomy, perch_labels, primary_labels):
    bc = (perch_labels.reset_index().rename(columns={"index": "bc_index"}))
    name_col = "inat2024_fsd50k" if "inat2024_fsd50k" in bc.columns else bc.columns[1]
    bc = bc.rename(columns={name_col: "scientific_name"})
    no_label = len(bc)
    m = taxonomy.merge(bc[["scientific_name", "bc_index"]], on="scientific_name", how="left")
    m["bc_index"] = m["bc_index"].fillna(no_label).astype(int)
    l2bc = m.set_index("primary_label")["bc_index"]
    bc_idx = np.array([int(l2bc.loc[c]) for c in primary_labels])
    mapped = bc_idx != no_label
    cls = taxonomy.set_index("primary_label")["class_name"].to_dict()
    proxies = {}
    for _, r in m[(m["bc_index"] == no_label)
                  & ~m["primary_label"].astype(str).str.contains("son")].iterrows():
        if cls.get(r["primary_label"]) not in PROXY_TAXA:
            continue
        genus = str(r["scientific_name"]).split()[0]
        hits = bc[bc["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\s", na=False)]
        if len(hits):
            proxies[primary_labels.index(r["primary_label"])] = hits["bc_index"].astype(int).values
    print(f"Mapped {mapped.sum()}/{len(primary_labels)} species, {len(proxies)} genus proxies")
    return bc_idx, mapped, proxies, cls


def map_scores(logits, bc_idx, mapped, proxies):
    s = np.zeros((len(logits), len(bc_idx)), np.float32)
    s[:, mapped] = logits[:, bc_idx[mapped]]
    for pos, idx in proxies.items():
        s[:, pos] = logits[:, idx].max(axis=1)
    return s


def smooth(scores, cols, alpha, mode):
    if alpha <= 0 or len(cols) == 0:
        return scores
    v = scores.reshape(-1, N_WINDOWS, scores.shape[1])
    x = v[:, :, cols]
    prev = np.concatenate([x[:, :1], x[:, :-1]], 1)
    nxt = np.concatenate([x[:, 1:], x[:, -1:]], 1)
    v[:, :, cols] = ((1 - alpha) * x + 0.5 * alpha * (prev + nxt)) if mode == "texture" \
        else ((1 - alpha) * x + alpha * np.maximum(x, np.maximum(prev, nxt)))
    return v.reshape(scores.shape)


def fuse(scores, sites, hours, tables, groups):
    s = scores.copy()
    prior = prior_logits(sites, hours, tables)
    g = groups
    s[:, g["mapped_event"]] += BEST["lambda_event"] * prior[:, g["mapped_event"]]
    s[:, g["mapped_texture"]] += BEST["lambda_texture"] * prior[:, g["mapped_texture"]]
    s[:, g["proxy_texture"]] += BEST["lambda_proxy_texture"] * prior[:, g["proxy_texture"]]
    s[:, g["prioronly_event"]] = BEST["lambda_event"] * prior[:, g["prioronly_event"]]
    s[:, g["prioronly_texture"]] = BEST["lambda_texture"] * prior[:, g["prioronly_texture"]]
    s[:, g["unmapped_inactive"]] = -8.0
    s = smooth(s, g["texture"], BEST["smooth_texture"], "texture")
    return smooth(s, g["event"], BEST["smooth_event"], "event")


def class_groups(primary_labels, cls, mapped, proxies, Y_sc):
    active = Y_sc.sum(0) > 0
    tex = np.array([cls.get(l) in TEXTURE_TAXA for l in primary_labels])
    prox = np.zeros(len(primary_labels), bool); prox[list(proxies)] = True
    w = lambda m: np.where(m)[0]
    return {
        "texture": w(active & tex), "event": w(active & ~tex),
        "mapped_event": w(active & ~tex & mapped), "mapped_texture": w(active & tex & mapped),
        "proxy_texture": w(active & tex & prox),
        "prioronly_event": w(active & ~tex & ~mapped & ~prox),
        "prioronly_texture": w(active & tex & ~mapped & ~prox),
        "unmapped_inactive": w(~active & ~mapped),
    }


# ------------------------------------------------------------------ main
def sc_file_folds(sc_cache_meta):
    """Same GroupKFold assignment as model1_sed.py (must stay in sync)."""
    files = sc_cache_meta[["filename", "site"]].drop_duplicates().reset_index(drop=True)
    files["fold"] = -1
    for f, (_, vi) in enumerate(GroupKFold(n_splits=N_FOLDS).split(files, groups=files["filename"])):
        files.loc[files.index[vi], "fold"] = f
    return dict(zip(files["filename"], files["fold"]))


def extract_chunk(w, start):
    """Same rule as model1_sed.extract_chunk_np."""
    if len(w) <= WIN:
        return np.pad(w, (WIN - len(w), 0)).astype(np.float32)
    start = min(start, len(w) - WIN)
    return w[start:start + WIN].astype(np.float32)


def batched_perch(items, load_fn, perch, flush_windows=512, workers=12, chunk=256):
    """Load items on `workers` threads (bounded: `chunk` at a time, in order) and run
    Perch on large batches of windows pooled across items, so the GPU is not left
    waiting on disk. load_fn(item) -> (n_windows, WIN) float32 array, or None to skip.
    Yields (item, windows, logits, emb, spatial) per item, in input order."""
    from concurrent.futures import ThreadPoolExecutor
    buf, n_buf = [], 0

    def flush():
        L, E, S = perch(np.concatenate([x for _, x in buf]))
        o = 0
        for it, x in buf:
            n = len(x)
            yield it, x, L[o:o + n], E[o:o + n], (None if S is None else S[o:o + n])
            o += n

    with ThreadPoolExecutor(workers) as ex:
        for c in range(0, len(items), chunk):
            for it, x in zip(items[c:c + chunk], ex.map(load_fn, items[c:c + chunk])):
                if x is None:
                    continue
                buf.append((it, x)); n_buf += len(x)
                if n_buf >= flush_windows:
                    yield from flush()
                    buf, n_buf = [], 0
        if buf:
            yield from flush()


def load_ogg(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(1)
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    y = y[:N_WINDOWS * WIN]
    return np.pad(y, (0, N_WINDOWS * WIN - len(y)))


def save_arrays(out, prefix, emb, spatial):
    np.save(out / f"{prefix}_emb.npy", np.concatenate(emb).astype(np.float16))
    if spatial and spatial[0] is not None:
        np.save(out / f"{prefix}_spatial.npy", np.concatenate(spatial).astype(np.float16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp-dir", required=True, type=Path)
    ap.add_argument("--waveform-cache", required=True, type=Path)
    ap.add_argument("--perch-onnx", required=True, type=Path)
    ap.add_argument("--perch-labels", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-windows-per-clip", type=int, default=6)
    ap.add_argument("--max-unlabeled-files", type=int, default=2000)
    ap.add_argument("--min-detect", type=float, default=0.05)
    ap.add_argument("--no-spatial", action="store_true")
    ap.add_argument("--skip-focal", action="store_true")
    ap.add_argument("--batch", type=int, default=128, help="Perch windows per GPU batch")
    ap.add_argument("--workers", type=int, default=12, help="threads loading/decoding audio")
    ap.add_argument("--allow-cpu", action="store_true", help="permit CPU fallback (slow)")
    args = ap.parse_args()
    t_start = time.time()
    out = args.out; (out / "sc_wave").mkdir(parents=True, exist_ok=True)

    primary = pd.read_csv(args.comp_dir / "sample_submission.csv").columns[1:].tolist()
    l2i = {l: i for i, l in enumerate(primary)}
    taxonomy = pd.read_csv(args.comp_dir / "taxonomy.csv", dtype={"primary_label": str})
    bc_idx, mapped, proxies, cls = build_mapping(taxonomy, pd.read_csv(args.perch_labels), primary)
    perch = Perch(args.perch_onnx, batch=args.batch, allow_cpu=args.allow_cpu)
    keep_spatial = not args.no_spatial and perch.spatial_key is not None
    if not keep_spatial:
        print("No spatial embedding saved -> the Perch probe will attend over 1 frame per window")

    # ---------------- focal clips
    if not args.skip_focal:
        meta = pd.read_csv(args.waveform_cache / "audio_cache_meta.csv", dtype={"primary_label": str})
        train_df = pd.read_csv(args.comp_dir / "train.csv", dtype={"primary_label": str})
        meta = meta.merge(train_df[["filename", "secondary_labels"]], on="filename", how="left")
        meta = meta[meta["primary_label"].astype(str).isin(l2i)].drop_duplicates("cache_file")
        rows, T, E, S = [], [], [], []

        def load_focal_windows(r):
            p = args.waveform_cache / r.cache_file
            if not p.exists():
                return None
            w = torch.load(p, map_location="cpu").float().numpy().reshape(-1) / 32767.0
            nw = int(min(args.max_windows_per_clip, max(1, np.ceil(len(w) / WIN))))
            # cut windows exactly like model1_sed.extract_chunk_np so Perch labels the same
            # audio the CNN trains on (short clips left-padded, last window shifted back)
            return np.stack([extract_chunk(w, 5 * k * SR) for k in range(nw)])

        items = list(meta.itertuples(index=False))
        for n, (r, x, logits, emb, sp) in enumerate(
                batched_perch(items, load_focal_windows, perch, workers=args.workers)):
            nw = len(x)
            prob = 1 / (1 + np.exp(-map_scores(logits, bc_idx, mapped, proxies)))
            labels = [str(r.primary_label)]
            try:
                labels += [s for s in eval(r.secondary_labels) if s in l2i]
            except Exception:
                pass
            tgt = np.zeros((nw, len(primary)), np.float32)
            for lab in labels:
                c = l2i[str(lab)]
                pmax = prob[:, c].max()
                tgt[:, c] = prob[:, c] / pmax if pmax >= args.min_detect else 1.0
            for k in range(nw):
                rows.append({"cache_file": r.cache_file, "filename": r.filename,
                             "original_idx": int(r.original_idx),
                             "primary_label": str(r.primary_label), "start_sec": 5 * k})
            T.append(tgt); E.append(emb); S.append(sp if keep_spatial else None)
            if n % 1000 == 0:
                print(f"focal {n}/{len(items)}  windows={len(rows)}  "
                      f"[{time.time() - t_start:.0f}s]", flush=True)
        pd.DataFrame(rows).to_parquet(out / "focal_windows.parquet", index=False)
        np.save(out / "focal_targets.npy", np.concatenate(T).astype(np.float16))
        save_arrays(out, "focal", E, S if keep_spatial else None)

    # ---------------- soundscapes
    sc_meta = pd.read_csv(args.waveform_cache / "soundscape_cache_meta.csv")
    labels_raw = pd.read_csv(args.comp_dir / "train_soundscapes_labels.csv", dtype={"primary_label": str}).drop_duplicates()
    labels_raw["start_sec"] = pd.to_timedelta(labels_raw["start"]).dt.total_seconds().astype(int)
    labeled_files = sorted(labels_raw["filename"].unique())
    all_files = sorted(p.name for p in (args.comp_dir / "train_soundscapes").glob("*.ogg"))
    unlabeled = [f for f in all_files if f not in set(labeled_files)][:args.max_unlabeled_files]
    file_fold = sc_file_folds(sc_meta)

    rows, SC, E, S = [], [], [], []
    sc_items = labeled_files + unlabeled
    load_sc = lambda fn: load_ogg(args.comp_dir / "train_soundscapes" / fn).reshape(N_WINDOWS, WIN)
    for n, (fn, x, logits, emb, sp) in enumerate(
            batched_perch(sc_items, load_sc, perch, workers=args.workers)):
        is_lab = fn in file_fold
        if not is_lab:
            np.save(out / "sc_wave" / f"{Path(fn).stem}.npy",
                    (x.reshape(-1) * 32767).clip(-32768, 32767).astype(np.int16))
        site, hour = site_hour(fn)
        for k in range(N_WINDOWS):
            rows.append({"filename": fn, "start_sec": 5 * k, "site": site, "hour_utc": hour,
                         "labeled": is_lab, "fold": file_fold.get(fn, -1)})
        SC.append(map_scores(logits, bc_idx, mapped, proxies)); E.append(emb)
        S.append(sp if keep_spatial else None)
        if n % 200 == 0:
            print(f"soundscape {n}/{len(sc_items)}", flush=True)
    win = pd.DataFrame(rows)
    scores = np.concatenate(SC)

    # ground truth for labelled windows
    Y = np.zeros((len(win), len(primary)), np.float32)
    key = {(f, s): i for i, (f, s) in enumerate(zip(win["filename"], win["start_sec"]))}
    for r in labels_raw.itertuples(index=False):
        i = key.get((r.filename, r.start_sec))
        if i is not None:
            for lab in str(r.primary_label).split(";"):
                if lab.strip() in l2i:
                    Y[i, l2i[lab.strip()]] = 1.0
    lab = win["labeled"].values
    groups = class_groups(primary, cls, mapped, proxies, Y[lab])

    for k in range(N_FOLDS):
        fit = lab & (win["fold"].values != k)
        tables = fit_prior_tables(win["site"].values[fit], win["hour_utc"].values[fit], Y[fit])
        fused = fuse(scores, win["site"].values, win["hour_utc"].values, tables, groups)
        tgt = np.where(lab[:, None], Y, 1 / (1 + np.exp(-fused)))
        np.save(out / f"sc_targets_fold{k}.npy", tgt.astype(np.float16))
    win.to_parquet(out / "sc_windows.parquet", index=False)
    save_arrays(out, "sc", E, S if keep_spatial else None)
    (out / "pseudo_label_cost.json").write_text(json.dumps(
        {"wall_seconds": time.time() - t_start, "gpu_hours": (time.time() - t_start) / 3600,
         "soundscape_windows": int(len(win))}, indent=1))
    print(f"Done: {len(win)} soundscape windows ({lab.sum()} labelled) -> {out}")


if __name__ == "__main__":
    main()
