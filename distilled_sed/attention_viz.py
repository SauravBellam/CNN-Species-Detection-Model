"""Render the tracked attention / class-activation heatmaps and zip them for download.

model1_sed.py tracks TRACK_N fixed validation windows per fold and saves, for
every epoch, the attention weights over time and the class-activation map
(classifier applied to each (freq, time) cell of the ResNet feature map) for
each window's true species, to OUT_DIR/attention/fold{k}.npz.

    python -m distilled_sed.attention_viz runs/ablations/full/attention

writes attention/png/*.png and OUT_DIR/attention_heatmaps.zip. Each PNG has:
  1. log-mel (channel 0) with the best-epoch CAM overlaid,
  2. attention over time for every epoch (how the attention moved in training),
  3. predicted probability of the true species per epoch.
"""
import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom


def _upsample(a, shape):
    return zoom(a, (shape[0] / a.shape[0], shape[1] / a.shape[1]), order=1)


def render_fold(npz_path, png_dir):
    d = np.load(npz_path)
    best = int(np.nanargmax(d["ns22_auc"])) if len(d["ns22_auc"]) else -1
    fold = Path(npz_path).stem
    for i in range(len(d["row_id"])):
        mel, cam = d["mel"][i], d["cam"][best, i]
        cam = 1 / (1 + np.exp(-_upsample(cam, mel.shape)))
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.6), gridspec_kw={"width_ratios": [2, 1.4, 1]})
        ax[0].imshow(mel, origin="lower", aspect="auto", cmap="gray")
        im = ax[0].imshow(cam, origin="lower", aspect="auto", cmap="inferno", alpha=0.45, vmin=0, vmax=1)
        ax[0].set(title=f"{d['row_id'][i]}  [{d['species'][i]}]  CAM @ epoch {best}",
                  xlabel="time frame", ylabel="mel bin")
        fig.colorbar(im, ax=ax[0], fraction=0.03)
        ta = d["time_att"][:, i]                            # (epochs, T')
        ax[1].imshow(ta, origin="lower", aspect="auto", cmap="viridis",
                     extent=[0, 5, -0.5, ta.shape[0] - 0.5])
        ax[1].set(title="attention over time", xlabel="seconds", ylabel="epoch")
        ax[2].plot(d["prob"][:, i], marker="o", ms=3)
        ax[2].axvline(best, ls="--", c="gray")
        ax[2].set(title="p(true species)", xlabel="epoch", ylim=(0, 1))
        fig.tight_layout()
        fig.savefig(png_dir / f"{fold}_{i:02d}_{d['species'][i]}.png", dpi=110)
        plt.close(fig)


def render_dir(att_dir):
    att_dir = Path(att_dir)
    png_dir = att_dir / "png"
    png_dir.mkdir(exist_ok=True)
    for f in sorted(att_dir.glob("fold*.npz")):
        render_fold(f, png_dir)
    zip_path = shutil.make_archive(str(att_dir.parent / "attention_heatmaps"), "zip", att_dir)
    print(f"Attention heatmaps -> {zip_path}")
    return zip_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("attention_dir")
    render_dir(ap.parse_args().attention_dir)
