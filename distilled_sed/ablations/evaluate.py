"""Score every ablation run on many metrics and check each hypothesis.

    python -m distilled_sed.ablations.evaluate --runs runs/ablations \
        --taxonomy /path/to/birdclef-2026/taxonomy.csv

Writes <runs>/ablation_table.csv (variant x metric), ablation_deltas.csv
(variant - full) and hypotheses.md (predicted vs observed direction).
Metrics use non-S22 soundscape windows, like the notebook's primary metric.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .configs import HYPOTHESES, POSTHOC_VARIANTS, TRAIN_VARIANTS

RARE_MAX, COMMON_MIN = 20, 100  # focal training clips per species


def load_run(path):
    d = dict(np.load(path, allow_pickle=False))
    has = ~np.isnan(d["blend"][:, 0]) & d["ns22"]
    order = np.lexsort((d["start_sec"][has], d["filename"][has]))  # file, then time
    sel = np.where(has)[0][order]
    return d, sel


def score(d, sel, p, taxon_classes):
    tc = d["train_counts"]
    return M.all_metrics(d["y"][sel], p, d["filename"][sel], taxon_classes,
                         rare_classes=np.where(tc < RARE_MAX)[0],
                         common_classes=np.where(tc >= COMMON_MIN)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/ablations")
    ap.add_argument("--taxonomy", required=True)
    args = ap.parse_args()
    runs = Path(args.runs)

    rows = {}
    taxon_classes = None
    for name in TRAIN_VARIANTS:
        f = runs / name / "oof.npz"
        if not f.exists():
            print(f"[missing] {name}")
            continue
        d, sel = load_run(f)
        if taxon_classes is None:
            tax = pd.read_csv(args.taxonomy)
            l2t = dict(zip(tax["primary_label"].astype(str), tax["class_name"].astype(str)))
            labels = d["labels"]
            taxon_classes = {t: np.array([i for i, l in enumerate(labels) if l2t.get(l) == t])
                             for t in ["Aves", "Amphibia", "Insecta", "Mammalia"]}
        rows[name] = score(d, sel, d["blend"][sel], taxon_classes)
        if name == "full":
            for pname, mode in POSTHOC_VARIANTS.items():
                p = (M.gauss_smooth(d["blend"][sel], d["filename"][sel]) if mode == "smooth"
                     else d[mode][sel])
                rows[pname] = score(d, sel, p, taxon_classes)

    table = pd.DataFrame(rows).T
    table.to_csv(runs / "ablation_table.csv")
    print(table.round(4).to_string())
    if "full" not in table.index:
        return
    deltas = table - table.loc["full"]
    deltas.to_csv(runs / "ablation_deltas.csv")

    lines = ["| variant | metric | predicted | Δ vs full | biggest-Δ metric | verdict |",
             "|---|---|---|---|---|---|"]
    for var, metric, pred, why in HYPOTHESES:
        if var not in deltas.index:
            continue
        dv = deltas.loc[var, metric]
        biggest = deltas.loc[var].abs().idxmax()
        if pred in ("drop", "small drop"):
            ok = dv < 0
        elif pred == "rise":
            ok = dv > 0
        else:
            ok = abs(dv) < 0.005
        lines.append(f"| {var} | {metric} | {pred} | {dv:+.4f} | {biggest} | "
                     f"{'supported' if ok else 'not supported'} |")
        lines.append(f"| | | | | | _{why}_ |")
    (runs / "hypotheses.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
