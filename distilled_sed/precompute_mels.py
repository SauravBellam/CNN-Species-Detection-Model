"""Precompute the log-mel spectrogram of every 5 s training/validation window, once.

Training then reads spectrograms straight from disk (memory-mapped) instead of
loading audio files and computing STFTs every batch. Run after pseudo_label.py:

    python -m distilled_sed.precompute_mels --waveform-cache $WAVEFORM_CACHE_DIR --pseudo-dir $PSEUDO_DIR

Writes (float16, log-mel in dB, 80 dB floor, rows aligned with the parquet tables):
  <pseudo-dir>/focal_mel.npy   (n focal windows, n_mels, 313)  <- focal_windows.parquet
  <pseudo-dir>/sc_mel.npy      (n soundscape windows, n_mels, 313) <- sc_windows.parquet
                               (labelled + pseudo-labelled windows; validation uses these too)

Windows are cut exactly as in pseudo_label.py / the training loader, so the
spectrogram, the Perch label and the Perch embedding all describe the same audio.
Size: n_mels x 313 x 2 bytes per window (~160 KB at 256 mels).
"""
import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .models import LogMel
from .pseudo_label import WIN, SR, extract_chunk

N_FRAMES = WIN // 512 + 1  # 313


def run(name, n, load_fn, out_path, logmel, n_mels, workers, batch=256):
    print(f"{name}: {n} windows -> {out_path} ({n * n_mels * N_FRAMES * 2 / 1e9:.1f} GB)", flush=True)
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n, n_mels, N_FRAMES))
    with ThreadPoolExecutor(workers) as ex:
        for s in range(0, n, batch):
            wav = np.stack(list(ex.map(load_fn, range(s, min(n, s + batch)))))
            with torch.no_grad():
                mel = logmel(torch.from_numpy(wav).cuda())[:, 0]
            out[s:s + len(wav)] = mel.cpu().numpy().astype(np.float16)
            if (s // batch) % 50 == 0:
                print(f"  {name} {s}/{n}", flush=True)
    out.flush()
    del out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--waveform-cache", type=Path, required=True)
    ap.add_argument("--pseudo-dir", type=Path, required=True)
    ap.add_argument("--n-mels", type=int, default=256)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "precompute_mels runs on the GPU"
    logmel = LogMel(n_mels=args.n_mels).cuda().eval()

    focal = pd.read_parquet(args.pseudo_dir / "focal_windows.parquet")
    sc = pd.read_parquet(args.pseudo_dir / "sc_windows.parquet")
    need = (len(focal) + len(sc)) * args.n_mels * N_FRAMES * 2
    free = shutil.disk_usage(args.pseudo_dir).free
    print(f"Need {need / 1e9:.1f} GB, free {free / 1e9:.1f} GB")
    if need > free * 0.95:
        raise SystemExit("Not enough disk: free space or rerun with --n-mels 128")

    # ---- focal windows (from the waveform cache)
    def load_focal(i):
        r = focal.iloc[i]
        w = torch.load(args.waveform_cache / r.cache_file, map_location="cpu").float().numpy().reshape(-1)
        return extract_chunk(w / 32767.0, int(r.start_sec) * SR)

    run("focal", len(focal), load_focal, args.pseudo_dir / "focal_mel.npy", logmel, args.n_mels, args.workers)

    # ---- soundscape windows: labelled files from the waveform cache, unlabelled from sc_wave/
    file_meta = pd.read_csv(args.waveform_cache / "soundscape_file_meta.csv")
    cache_of = dict(zip(file_meta["filename"], file_meta["cache_file"]))

    def load_sc(i):
        r = sc.iloc[i]
        s0 = int(r.start_sec) * SR
        if r.labeled and r.filename in cache_of:
            w = torch.load(args.waveform_cache / cache_of[r.filename], map_location="cpu").numpy().reshape(-1)
        else:
            w = np.load(args.pseudo_dir / "sc_wave" / f"{Path(r.filename).stem}.npy", mmap_mode="r")
        return extract_chunk(np.asarray(w[s0:s0 + WIN], np.float32) / 32767.0, 0)

    run("soundscape", len(sc), load_sc, args.pseudo_dir / "sc_mel.npy", logmel, args.n_mels, args.workers)
    print("Done.")


if __name__ == "__main__":
    main()
