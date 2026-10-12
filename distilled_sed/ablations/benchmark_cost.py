"""Inference compute and time per prediction for every trained variant.

    python -m distilled_sed.ablations.benchmark_cost --runs runs/ablations --perch-model perch_v2.onnx

Run it AFTER training, when nothing else is using the machine -- timings
taken while other runs share the CPU/GPU are meaningless. For every run with
a config.json it rebuilds that variant's features + network (weights do not
change speed) and measures, per 5 s window ("one prediction"):

  params_m, size_mb        model size (fp32)
  gflops                   network FLOPs (torch FlopCounter; STFT excluded)
  cpu1_ms, cpu1_feat_ms    CPU latency with 1 thread (the cheap on-site box), total and features
  cpu4_ms                  CPU latency with 4 threads
  gpu_ms                   GPU latency at batch 64 (if a GPU is present)
  preds_per_sec_cpu1       1000 / cpu1_ms
  realtime_x_cpu1          seconds of audio processed per second of compute

perch_attn_probe is timed as Perch v2 + the probe head on the GPU only (Perch is
never run on the CPU in this project, so its CPU columns are empty). Results go to
<runs>/cost.csv, which evaluate.py and plot.py merge with the quality metrics.
Without --runs it benchmarks a fixed set of reference configs instead.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from ..models import BirdSEDModel, MelFeatures

WIN = 160000


def timed(fn, x, iters, cuda=False):
    with torch.inference_mode():
        for _ in range(3):
            fn(x)
        if cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            out = fn(x)
        if cuda:
            torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters, out


def flops_per_window(model, x):
    with torch.inference_mode(), FlopCounterMode(display=False) as fc:
        model(x[:1])
    return fc.get_total_flops() / 1e9


def bench_sed(cfg, args):
    feat = MelFeatures(n_fft=cfg.get("N_FFT", 2048), hop_length=cfg.get("HOP_LENGTH", 512),
                       n_mels=cfg.get("N_MELS", 256), stack_mode=cfg.get("STACK_MODE", "delta")).eval()
    model = BirdSEDModel(num_classes=cfg.get("NUM_CLASSES", 234), pretrained=False,
                         in_chans=feat.n_channels, head=cfg.get("HEAD", "sed"), use_distill=False,
                         freq_pool=cfg.get("FREQ_POOL", "gem"), time_pool=cfg.get("TIME_POOL", "att")).eval()
    n = sum(p.numel() for p in model.parameters())
    wav = torch.randn(args.batch, 1, WIN)
    r = {"params_m": n / 1e6, "size_mb": n * 4 / 1e6}
    for threads in (1, 4):
        torch.set_num_threads(threads)
        tf, x = timed(feat, wav, args.iters)
        tn, _ = timed(model, x, args.iters)
        r[f"cpu{threads}_feat_ms"] = tf / args.batch * 1000
        r[f"cpu{threads}_ms"] = (tf + tn) / args.batch * 1000
    r["gflops"] = flops_per_window(model, x)
    if torch.cuda.is_available():
        fg, mg = feat.cuda(), model.cuda()
        wg = torch.randn(64, 1, WIN, device="cuda")
        t, _ = timed(lambda w: mg(fg(w)), wg, args.iters, cuda=True)
        r["gpu_ms"] = t / 64 * 1000
    return r


def bench_perch_probe(cfg, args):
    """Perch runs only on the GPU, so the probe is timed on the GPU only; its CPU
    columns are left empty (Perch is never run on the CPU in this project)."""
    from ..perch_gpu import PerchGPU
    from ..perch_probe import AttnPoolProbe
    probe = AttnPoolProbe(cfg["dim"], cfg["NUM_CLASSES"]).eval().cuda()
    n = sum(p.numel() for p in probe.parameters())
    perch = PerchGPU(args.perch_model, batch=64)
    x = np.random.randn(64, WIN).astype(np.float32)
    feats = torch.randn(64, cfg["frames"], cfg["dim"], device="cuda")
    tp, _ = timed(perch, x, args.iters)
    th, _ = timed(probe, feats, args.iters, cuda=True)
    nan = float("nan")
    return {"params_m": n / 1e6, "size_mb": n * 4 / 1e6, "gflops": nan,
            "cpu1_ms": nan, "cpu1_feat_ms": nan, "cpu4_ms": nan, "cpu4_feat_ms": nan,
            "gpu_ms": (tp + th) / 64 * 1000,
            "note": "Perch is GPU-only; params/size are the probe head only"}


REFERENCE = {  # used when --runs is not given
    "full": {"HEAD": "sed", "STACK_MODE": "delta"},
    "single_channel_mel": {"HEAD": "sed", "STACK_MODE": "single"},
    "multires_mel": {"HEAD": "sed", "STACK_MODE": "multires"},
    "delta_128mel": {"HEAD": "sed", "STACK_MODE": "delta", "N_MELS": 128},
    "resnet_baseline": {"HEAD": "pooled", "STACK_MODE": "delta"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None, help="benchmark every run dir with a config.json")
    ap.add_argument("--perch-model", "--perch-onnx", dest="perch_model", default=None,
                    help="Perch v2 .onnx or GPU SavedModel dir (timed on GPU only)")
    ap.add_argument("--batch", type=int, default=12)  # one 60 s file
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    if args.runs:
        configs = {p.parent.name: json.loads(p.read_text())
                   for p in sorted(Path(args.runs).glob("*/config.json"))}
    else:
        configs = {k: {"kind": "sed", **v} for k, v in REFERENCE.items()}

    rows = []
    for name, cfg in configs.items():
        if cfg.get("kind") == "perch_probe":
            if not args.perch_model or not torch.cuda.is_available():
                print(f"[skip] {name}: Perch is GPU-only; needs --perch-model and a GPU")
                continue
            r = bench_perch_probe(cfg, args)
        else:
            r = bench_sed(cfg, args)
        r["variant"] = name
        r["preds_per_sec_cpu1"] = 1000 / r["cpu1_ms"]
        r["realtime_x_cpu1"] = 5000 / r["cpu1_ms"]
        if r.get("gpu_ms"):
            r["preds_per_sec_gpu"] = 1000 / r["gpu_ms"]
        rows.append(r)
        print(f"{name:24s} params {r['params_m']:6.2f}M  GFLOPs {r['gflops']:6.2f}  "
              f"CPU1 {r['cpu1_ms']:7.1f} ms  CPU4 {r['cpu4_ms']:7.1f} ms  "
              f"GPU {r.get('gpu_ms', float('nan')):6.2f} ms  ({r['realtime_x_cpu1']:.0f}x realtime)",
              flush=True)

    df = pd.DataFrame(rows).set_index("variant")
    out = args.csv or (Path(args.runs) / "cost.csv" if args.runs else None)
    if out:
        df.to_csv(out)
        print(f"Cost -> {out}")


if __name__ == "__main__":
    main()
