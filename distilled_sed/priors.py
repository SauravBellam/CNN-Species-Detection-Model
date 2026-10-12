"""Site / hour prior tables and taxonomy smoothing, from nina2025's Kaggle
notebook "BirdCLEF 2026 EoS.9" (Model_22 cell 7 and the TAX_SMOOTHING blend
step). Shared by pseudo_label.py and the post-hoc ablations.

Both are nearly free at inference: the prior is a table lookup by
(recording site, hour of day), taxonomy smoothing is a few averages.
"""
import re

import numpy as np

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


def site_hour(filename):
    m = FNAME_RE.match(str(filename))
    return (m.group(2), int(m.group(4)[:2])) if m else ("", -1)


def fit_prior_tables(sites, hours, Y):
    """Species frequency per site, per hour and per (site, hour) from labelled windows."""
    sites, hours = np.asarray(sites), np.asarray(hours)
    t = {"global_p": Y.mean(0).astype(np.float32)}
    for key, vals in (("site", sites), ("hour", hours)):
        keys = sorted(set(vals.tolist()))
        t[f"{key}_to_i"] = {k: i for i, k in enumerate(keys)}
        t[f"{key}_n"] = np.array([(vals == k).sum() for k in keys], np.float32)
        t[f"{key}_p"] = np.stack([Y[vals == k].mean(0) for k in keys]).astype(np.float32)
    pairs = list(zip(sites.tolist(), hours.tolist()))
    keys = sorted(set(pairs))
    t["sh_to_i"] = {k: i for i, k in enumerate(keys)}
    idx = np.array([t["sh_to_i"][p] for p in pairs])
    t["sh_n"] = np.bincount(idx, minlength=len(keys)).astype(np.float32)
    t["sh_p"] = np.stack([Y[idx == i].mean(0) for i in range(len(keys))]).astype(np.float32)
    return t


def prior_logits(sites, hours, t, eps=1e-4):
    """Shrunk prior: global -> hour (k=8) -> site (k=8) -> site-hour (k=4), as logits."""
    n = len(sites)
    p = np.repeat(t["global_p"][None], n, 0)
    for key, vals, k in (("hour", hours, 8.0), ("site", sites, 8.0)):
        idx = np.array([t[f"{key}_to_i"].get(v, -1) for v in vals])
        ok = idx >= 0
        w = t[f"{key}_n"][idx[ok]][:, None] / (t[f"{key}_n"][idx[ok]][:, None] + k)
        p[ok] = w * t[f"{key}_p"][idx[ok]] + (1 - w) * p[ok]
    idx = np.array([t["sh_to_i"].get((s, h), -1) for s, h in zip(sites, hours)])
    ok = idx >= 0
    w = t["sh_n"][idx[ok]][:, None] / (t["sh_n"][idx[ok]][:, None] + 4.0)
    p[ok] = w * t["sh_p"][idx[ok]] + (1 - w) * p[ok]
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def taxonomy_smooth(p, labels, taxonomy, genus_alpha=0.15, class_alpha=0.05):
    """Pull each species toward its genus mean, then its class mean (multi-member groups only)."""
    sci = dict(zip(taxonomy["primary_label"].astype(str), taxonomy["scientific_name"].astype(str)))
    cls = dict(zip(taxonomy["primary_label"].astype(str), taxonomy["class_name"].astype(str)))
    p = p.copy()
    for key, alpha in ((lambda l: sci.get(l, l).split(" ")[0], genus_alpha),
                       (lambda l: cls.get(l, ""), class_alpha)):
        groups = {}
        for i, l in enumerate(labels):
            g = key(str(l))
            if g:
                groups.setdefault(g, []).append(i)
        for idx in groups.values():
            if len(idx) > 1:
                p[:, idx] = (1 - alpha) * p[:, idx] + alpha * p[:, idx].mean(1, keepdims=True)
    return p
