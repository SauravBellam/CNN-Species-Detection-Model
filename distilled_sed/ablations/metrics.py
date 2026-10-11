"""Metrics beyond ROC-AUC for the ablation study.

All functions take y (N, C) binary labels and p (N, C) probabilities.
Classes with no positives (or no negatives) in the evaluated rows are skipped
for per-class metrics, matching the competition's macro-AUC convention.
"""
import numpy as np
from scipy.ndimage import convolve1d
from sklearn.metrics import (average_precision_score, label_ranking_average_precision_score,
                             roc_auc_score)


def _valid_classes(y):
    pos = y.sum(0)
    return np.where((pos > 0) & (pos < len(y)))[0]


def macro_auc(y, p, classes=None):
    cls = _valid_classes(y) if classes is None else np.intersect1d(classes, _valid_classes(y))
    return float(np.mean([roc_auc_score(y[:, c], p[:, c]) for c in cls])) if len(cls) else np.nan


def macro_ap(y, p):
    cls = _valid_classes(y)
    return float(np.mean([average_precision_score(y[:, c], p[:, c]) for c in cls]))


def lrap(y, p):
    keep = y.sum(1) > 0
    return float(label_ranking_average_precision_score(y[keep], p[keep]))


def f1_macro(y, p, thr=0.5):
    cls = _valid_classes(y)
    pred = p >= thr
    tp = (pred & (y > 0)).sum(0)[cls]
    fp = (pred & (y == 0)).sum(0)[cls]
    fn = (~pred & (y > 0)).sum(0)[cls]
    return float(np.mean(2 * tp / np.maximum(2 * tp + fp + fn, 1)))


def ece(y, p, n_bins=15):
    """Expected calibration error over all (window, class) cells."""
    y, p = y.ravel(), p.ravel()
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    err = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def temporal_jitter(p, filenames):
    """Mean |p_t - p_{t-1}| between consecutive windows of the same file."""
    same = filenames[1:] == filenames[:-1]
    return float(np.abs(np.diff(p, axis=0))[same].mean())


def gauss_smooth(p, filenames, kernel=(0.1, 0.2, 0.4, 0.2, 0.1)):
    """The notebook's inference smoothing, applied per file in logit space."""
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    out = logit.copy()
    for f in np.unique(filenames):
        m = filenames == f
        out[m] = convolve1d(logit[m], np.asarray(kernel), axis=0, mode="nearest")
    return 1 / (1 + np.exp(-out))


def all_metrics(y, p, filenames, taxon_classes, rare_classes, common_classes):
    r = {
        "auc_soundscape_macro": macro_auc(y, p),
        "pr_auc_macro": macro_ap(y, p),
        "lrap": lrap(y, p),
        "f1_macro": f1_macro(y, p),
        "ece": ece(y, p),
        "brier": brier(y, p),
        "temporal_jitter": temporal_jitter(p, filenames),
        "auc_rare": macro_auc(y, p, rare_classes),
        "auc_common": macro_auc(y, p, common_classes),
    }
    for t, cls in taxon_classes.items():
        r[f"auc_{t}"] = macro_auc(y, p, cls)
    return r
