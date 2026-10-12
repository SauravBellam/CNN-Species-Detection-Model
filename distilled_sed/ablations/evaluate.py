"""Score every ablation run on many metrics and check each hypothesis.

    python -m distilled_sed.ablations.evaluate --runs runs/ablations \
        --taxonomy /path/to/birdclef-2026/taxonomy.csv

Writes <runs>/ablation_table.csv (variant x metric), ablation_deltas.csv
(variant - full) and hypotheses.md (predicted vs observed direction).
Metrics use non-S22 soundscape windows, like the notebook's primary metric.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from ..priors import fit_prior_tables, prior_logits, site_hour, taxonomy_smooth
from .configs import HYPOTHESES, POSTHOC_VARIANTS, TRAIN_VARIANTS

RARE_MAX, COMMON_MIN = 20, 100  # focal training clips per species


def load_run(path):
    d = dict(np.load(path, allow_pickle=False))
    has = ~np.isnan(d["blend"][:, 0]) & d["ns22"]
    order = np.lexsort((d["start_sec"][has], d["filename"][has]))  # file, then time
    sel = np.where(has)[0][order]
    return d, sel


TEXTURE_TAXA = {"Amphibia", "Insecta"}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def add_site_hour_prior(d, sel, p, lam):
    """Fold-safe: tables for fold k are fit only on the other folds' labelled windows.
    lam is per class (1.0 frogs/insects, 0.4 others, as in the notebook)."""
    sh = [site_hour(f) for f in d["filename"]]
    sites = np.array([a for a, _ in sh]); hours = np.array([b for _, b in sh])
    z = _logit(p)
    folds = d["fold"][sel]
    for k in np.unique(folds):
        fit = d["fold"] != k
        t = fit_prior_tables(sites[fit], hours[fit], d["y"][fit])
        m = folds == k
        z[m] += lam * prior_logits(sites[sel][m], hours[sel][m], t)
    return 1 / (1 + np.exp(-z))


def posthoc(mode, d, sel, tax, lam):
    p = d["blend"][sel]
    if mode in ("clip", "fmax"):
        return d[mode][sel]
    if mode in ("prior", "all"):
        p = add_site_hour_prior(d, sel, p, lam)
    if mode in ("smooth", "all"):
        p = M.gauss_smooth(p, d["filename"][sel])
    if mode in ("tax", "all"):
        p = taxonomy_smooth(p, d["labels"], tax)
    return p


def bang_for_buck(table, runs):
    """Quality x cost per variant. Post-hoc variants cost the same as `full` (their extra
    work -- a table lookup, a few averages -- is negligible next to the network)."""
    cost = pd.read_csv(runs / "cost.csv", index_col=0)
    rows = []
    for v in table.index:
        base = v if v in cost.index else "full"
        if base not in cost.index:
            continue
        r = {"variant": v, **table.loc[v, ["auc_soundscape_macro", "pr_auc_macro", "lrap",
                                           "f1_macro", "ece", "auc_rare"]].to_dict(),
             **cost.loc[base].drop(labels=["note"], errors="ignore").to_dict()}
        tc = runs / (v if (runs / v / "train_cost.json").exists() else "full") / "train_cost.json"
        if tc.exists():
            t = json.loads(tc.read_text())
            r.update(train_gpu_hours=t.get("gpu_hours_est"), peak_gpu_mem_gb=t.get("peak_gpu_mem_gb"),
                     mean_epoch_seconds=t.get("mean_epoch_seconds"))
        rows.append(r)
    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("variant").sort_values("cpu1_ms")
    # Pareto frontier: no other variant is both cheaper (CPU, 1 thread) and better (macro AUC)
    q, c = df["auc_soundscape_macro"].values, df["cpu1_ms"].values
    df["pareto"] = [not np.any((c <= c[i]) & (q >= q[i]) & ((c < c[i]) | (q > q[i])))
                    for i in range(len(df))]
    if "resnet_baseline" in df.index:
        b = df.loc["resnet_baseline"]
        extra_ms = df["cpu1_ms"] - b["cpu1_ms"]
        df["auc_gain_vs_baseline"] = df["auc_soundscape_macro"] - b["auc_soundscape_macro"]
        df["auc_gain_per_extra_ms"] = np.where(extra_ms.abs() > 1e-3,
                                               df["auc_gain_vs_baseline"] / extra_ms, np.nan)
    df.to_csv(runs / "bang_for_buck.csv")
    cols = [c for c in ["auc_soundscape_macro", "cpu1_ms", "gflops", "params_m",
                        "train_gpu_hours", "auc_gain_per_extra_ms", "pareto"] if c in df.columns]
    (runs / "bang_for_buck.md").write_text(df[cols].round(4).to_markdown() + "\n")
    print("\nBang for buck (sorted by CPU ms per prediction):")
    print(df[cols].round(4).to_string())
    return df


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
            lam = np.array([1.0 if l2t.get(str(l)) in TEXTURE_TAXA else 0.4 for l in d["labels"]])
            for pname, mode in POSTHOC_VARIANTS.items():
                rows[pname] = score(d, sel, posthoc(mode, d, sel, tax, lam), taxon_classes)

    table = pd.DataFrame(rows).T
    table.to_csv(runs / "ablation_table.csv")
    print(table.round(4).to_string())
    if "full" not in table.index:
        return
    if (runs / "cost.csv").exists():
        bang_for_buck(table, runs)
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
