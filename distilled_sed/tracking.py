"""Per-epoch metric tracking shared by every run (CNN variants and the Perch probe).

Each run writes <run>/curves.csv in long format so ablations/plot.py can draw
learning curves for any metric:

    variant, scope, epoch, metric, value
    full,    fold0, 3,     lrap,   0.71
    full,    pooled,3,     val_loss, 0.032

scope is "fold<k>" (that fold's validation windows) or "pooled" (all folds'
validation windows for that epoch together -- the curve the paper reports).
Metrics: everything in ablations/metrics.all_metrics plus train/val loss.
"""
import numpy as np
import pandas as pd

from .ablations.metrics import all_metrics

RARE_MAX, COMMON_MIN = 20, 100  # focal training clips per species


def class_groups(labels, label_to_taxon, train_counts):
    taxa = {t: np.array([i for i, l in enumerate(labels) if label_to_taxon.get(str(l)) == t])
            for t in ["Aves", "Amphibia", "Insecta", "Mammalia"]}
    tc = np.asarray(train_counts)
    return taxa, np.where(tc < RARE_MAX)[0], np.where(tc >= COMMON_MIN)[0]


def eval_metrics(y, p, filenames, start_secs, ns22, groups):
    """Metrics on non-S22 windows, ordered by file then time (needed for jitter)."""
    keep = np.where(ns22 & ~np.isnan(p[:, 0]))[0]
    keep = keep[np.lexsort((np.asarray(start_secs)[keep], np.asarray(filenames)[keep]))]
    y, p, fn = y[keep], p[keep], np.asarray(filenames)[keep]
    taxa, rare, common = groups
    r = all_metrics(y, p, fn, taxa, rare, common)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    r["val_loss"] = float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean())
    return r


def to_long(variant, scope, epoch, metrics):
    return [{"variant": variant, "scope": scope, "epoch": epoch, "metric": k, "value": float(v)}
            for k, v in metrics.items()]


def write_curves(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Per-epoch curves -> {path}")
