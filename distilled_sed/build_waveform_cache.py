"""Build the waveform cache Model_1 expects from the raw competition audio.

The notebook read a pre-built Kaggle dataset (tuckerarrants/birdclef-2026-
waveform-cache). This script recreates the same layout locally:

  <out>/audio_cache_meta.csv      cache_file, filename, primary_label, original_idx
  <out>/soundscape_cache_meta.csv filename, cache_file, start_sec, site, label_list
                                  (all 12 windows of every labelled soundscape)
  <out>/soundscape_file_meta.csv  filename, cache_file
  <out>/focal/*.pt, <out>/soundscapes/*.pt   int16 torch tensors at 32 kHz mono

Focal clips are capped at --max-seconds (default 30 s = the 6 windows
pseudo_label.py uses) to bound disk use: int16 at 32 kHz is 64 KB per second.

    python -m distilled_sed.build_waveform_cache --comp-dir $COMP_DIR --out $WAVEFORM_CACHE_DIR
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from .priors import site_hour

SR = 32000


def load(path, max_seconds=None):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(1)
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if max_seconds:
        y = y[:int(max_seconds * SR)]
    return torch.from_numpy((np.clip(y, -1, 1) * 32767).astype(np.int16))


def _save(job):
    src, dst, max_s = job
    if dst.exists():
        return True
    try:
        torch.save(load(src, max_s), dst)
        return True
    except Exception as e:  # unreadable file: skip it, training treats it as missing
        print(f"skip {src}: {e}", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    (args.out / "focal").mkdir(parents=True, exist_ok=True)
    (args.out / "soundscapes").mkdir(parents=True, exist_ok=True)

    # ---- focal clips
    train = pd.read_csv(args.comp_dir / "train.csv", dtype={"primary_label": str})
    train["original_idx"] = np.arange(len(train))
    train["cache_file"] = [f"focal/{i:06d}.pt" for i in train["original_idx"]]
    jobs = [(args.comp_dir / "train_audio" / fn, args.out / cf, args.max_seconds)
            for fn, cf in zip(train["filename"], train["cache_file"])]
    print(f"Focal clips: {len(jobs)} (<= {args.max_seconds:.0f} s each, "
          f"<= {len(jobs) * args.max_seconds * SR * 2 / 1e9:.0f} GB)", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        ok = list(ex.map(_save, jobs, chunksize=64))
    train[np.array(ok)][["cache_file", "filename", "primary_label", "original_idx"]] \
        .to_csv(args.out / "audio_cache_meta.csv", index=False)

    # ---- labelled soundscapes (full 60 s files)
    lab = pd.read_csv(args.comp_dir / "train_soundscapes_labels.csv", dtype={"primary_label": str}).drop_duplicates()
    lab["start_sec"] = pd.to_timedelta(lab["start"]).dt.total_seconds().astype(int)
    files = sorted(lab["filename"].unique())
    file_meta = pd.DataFrame({"filename": files,
                              "cache_file": [f"soundscapes/{Path(f).stem}.pt" for f in files]})
    jobs = [(args.comp_dir / "train_soundscapes" / fn, args.out / cf, 60)
            for fn, cf in zip(file_meta["filename"], file_meta["cache_file"])]
    with ProcessPoolExecutor(args.workers) as ex:
        ok = list(ex.map(_save, jobs))
    file_meta = file_meta[np.array(ok)]
    file_meta.to_csv(args.out / "soundscape_file_meta.csv", index=False)

    labels = lab.groupby(["filename", "start_sec"])["primary_label"].apply(
        lambda s: ";".join(sorted({l.strip() for x in s for l in str(x).split(";")})))
    rows = []
    for fn, cf in zip(file_meta["filename"], file_meta["cache_file"]):
        site = site_hour(fn)[0]
        for start in range(0, 60, 5):
            rows.append({"filename": fn, "cache_file": cf, "start_sec": start, "site": site,
                         "label_list": labels.get((fn, start), "")})
    pd.DataFrame(rows).to_csv(args.out / "soundscape_cache_meta.csv", index=False)
    print(f"Done: {sum(ok)} soundscape files, {len(rows)} windows -> {args.out}")


if __name__ == "__main__":
    main()
