"""Graphs from the tracked metrics of every run.

    python -m distilled_sed.ablations.plot --runs runs/ablations

Reads <runs>/*/curves.csv (written every epoch by model1_sed.py and
perch_probe.py), <runs>/ablation_table.csv (evaluate.py) and <runs>/cost.csv
(benchmark_cost.py), and writes to <runs>/plots/:

  all_curves.csv               every run's per-epoch metrics in one tidy table
  learning_curves.png          one panel per metric, pooled over folds; full model and
                               the two baselines highlighted, other ablations in gray
  curves_<variant>.png         one variant: every metric, each fold in gray + pooled
  final_<metric>.png           final OOF value per variant, sorted (incl. post-hoc ones)
  cost_vs_quality.png          macro ROC-AUC vs CPU ms per prediction, GFLOPs and training
                               GPU-hours, with the Pareto frontier (from bang_for_buck.csv)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reference categorical slots 1-3 (validated together); everything else is context gray.
HIGHLIGHT = {"full": "#2a78d6", "resnet_baseline": "#eb6834", "perch_attn_probe": "#1baf7a"}
CONTEXT = "#c3c2b7"
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e7e6e2"

METRICS = [  # (column, title, lower is better)
    ("train_loss", "Train loss", True), ("val_loss", "Validation loss (BCE)", True),
    ("auc_soundscape_macro", "Macro ROC-AUC", False), ("pr_auc_macro", "Macro PR-AUC", False),
    ("lrap", "LRAP", False), ("f1_macro", "Macro F1 @ 0.5", False),
    ("ece", "Calibration error (ECE)", True), ("brier", "Brier score", True),
    ("temporal_jitter", "Temporal jitter", True),
    ("auc_rare", "ROC-AUC, rare species (<20 clips)", False),
    ("auc_common", "ROC-AUC, common species (≥100 clips)", False),
    ("auc_Aves", "ROC-AUC, birds", False), ("auc_Amphibia", "ROC-AUC, amphibians", False),
    ("auc_Insecta", "ROC-AUC, insects", False), ("auc_Mammalia", "ROC-AUC, mammals", False),
]


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def learning_curves(df, out):
    pooled = df[df["scope"] == "pooled"]
    metrics = [m for m in METRICS if m[0] in set(pooled["metric"])]
    cols = 3
    rows = -(-len(metrics) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.6, rows * 3.0), squeeze=False)
    variants = sorted(pooled["variant"].unique(), key=lambda v: v in HIGHLIGHT)  # highlights on top
    for ax, (col, title, _) in zip(axes.flat, metrics):
        d = pooled[pooled["metric"] == col]
        for v in variants:
            s = d[d["variant"] == v].sort_values("epoch")
            if s.empty:
                continue
            hi = v in HIGHLIGHT
            ax.plot(s["epoch"], s["value"], color=HIGHLIGHT.get(v, CONTEXT),
                    lw=2 if hi else 1, zorder=3 if hi else 2)
            if hi:  # direct label at the line end
                ax.annotate(v, (s["epoch"].iloc[-1], s["value"].iloc[-1]), xytext=(4, 0),
                            textcoords="offset points", fontsize=7, color=INK, va="center")
        style(ax, title)
        ax.set_xlabel("epoch", fontsize=8, color=MUTED)
    for ax in axes.flat[len(metrics):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=c, lw=2, label=v) for v, c in HIGHLIGHT.items()]
    handles.append(plt.Line2D([], [], color=CONTEXT, lw=1, label="other ablations"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "learning_curves.png", dpi=130)
    plt.close(fig)


def variant_curves(df, out):
    for v, dv in df.groupby("variant"):
        metrics = [m for m in METRICS if m[0] in set(dv["metric"])]
        cols = 3
        rows = -(-len(metrics) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.8), squeeze=False)
        for ax, (col, title, _) in zip(axes.flat, metrics):
            d = dv[dv["metric"] == col]
            for scope, s in d.groupby("scope"):
                s = s.sort_values("epoch")
                pooled = scope == "pooled"
                ax.plot(s["epoch"], s["value"], color="#2a78d6" if pooled else CONTEXT,
                        lw=2 if pooled else 1, zorder=3 if pooled else 2)
            style(ax, title)
        for ax in axes.flat[len(metrics):]:
            ax.axis("off")
        fig.legend(handles=[plt.Line2D([], [], color="#2a78d6", lw=2, label="pooled over folds"),
                            plt.Line2D([], [], color=CONTEXT, lw=1, label="single fold")],
                   loc="upper center", ncol=2, frameon=False, fontsize=9)
        fig.suptitle(v, x=0.01, ha="left", y=0.995, fontsize=11, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out / f"curves_{v}.png", dpi=110)
        plt.close(fig)


def final_bars(table, out):
    for col, title, lower in METRICS:
        if col not in table.columns:
            continue
        s = table[col].dropna().sort_values(ascending=lower)
        fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(s) + 1.0))
        y = np.arange(len(s))[::-1]
        colors = [HIGHLIGHT.get(v, "#9ec5f4") for v in s.index]
        ax.barh(y, s.values, color=colors, height=0.6)
        ax.set_yticks(y, s.index, fontsize=8)
        if "full" in s.index:
            ax.axvline(s["full"], color=MUTED, lw=1, ls="--")
        lo, hi = s.min(), s.max()
        pad = (hi - lo) * 0.15 or 0.01
        ax.set_xlim(lo - pad, hi + pad * 2)
        for yi, val in zip(y, s.values):
            ax.text(val, yi, f" {val:.3f}", va="center", fontsize=7, color=INK)
        style(ax, f"{title} ({'lower' if lower else 'higher'} is better); dashed = full")
        ax.grid(axis="x", color=GRID, lw=0.8); ax.grid(axis="y", visible=False)
        fig.tight_layout()
        fig.savefig(out / f"final_{col}.png", dpi=130)
        plt.close(fig)


def cost_vs_quality(bfb, out):
    """Macro AUC against three costs; the line joins the Pareto-optimal variants."""
    costs = [(c, t, log) for c, t, log in [
        ("cpu1_ms", "CPU ms per prediction (1 thread)", True),
        ("gflops", "GFLOPs per prediction", True),
        ("train_gpu_hours", "Training GPU-hours (est.)", True)] if c in bfb and bfb[c].notna().any()]
    fig, axes = plt.subplots(1, len(costs), figsize=(5.2 * len(costs), 4.4), squeeze=False)
    for ax, (col, title, log) in zip(axes[0], costs):
        d = bfb[[col, "auc_soundscape_macro"]].dropna()
        d = d[d[col] > 0]
        for v, (c, q) in d.iterrows():
            hi = v in HIGHLIGHT
            ax.scatter(c, q, s=60 if hi else 30, color=HIGHLIGHT.get(v, CONTEXT),
                       edgecolor="white", linewidth=2, zorder=3 if hi else 2)
        front = d.sort_values(col)
        front = front[front["auc_soundscape_macro"] >= front["auc_soundscape_macro"].cummax()]
        ax.plot(front[col], front["auc_soundscape_macro"], color=MUTED, lw=1, ls="--", zorder=1)
        for v, (c, q) in front.iterrows():
            ax.annotate(v, (c, q), xytext=(5, 3), textcoords="offset points", fontsize=7, color=INK)
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(title, fontsize=9, color=MUTED)
        style(ax, f"Macro ROC-AUC vs {title.split(' (')[0]}")
    fig.suptitle("Bang for buck: up and left is better; dashed line = Pareto frontier (labelled)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "cost_vs_quality.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/ablations")
    runs = Path(ap.parse_args().runs)
    out = runs / "plots"
    out.mkdir(parents=True, exist_ok=True)

    curves = [pd.read_csv(f) for f in sorted(runs.glob("*/curves.csv"))]
    if curves:
        df = pd.concat(curves, ignore_index=True)
        df.to_csv(out / "all_curves.csv", index=False)
        learning_curves(df, out)
        variant_curves(df, out)
    if (runs / "ablation_table.csv").exists():
        table = pd.read_csv(runs / "ablation_table.csv", index_col=0)
        final_bars(table, out)
    if (runs / "bang_for_buck.csv").exists():
        cost_vs_quality(pd.read_csv(runs / "bang_for_buck.csv", index_col=0), out)
    print(f"Plots -> {out}")


if __name__ == "__main__":
    main()
