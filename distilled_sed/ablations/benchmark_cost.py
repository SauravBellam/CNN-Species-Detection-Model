"""Inference cost on CPU (the 24/7 on-site setting). No data needed.

    python -m distilled_sed.ablations.benchmark_cost --threads 1 [--perch-onnx perch_v2.onnx]

For each config: parameters, fp32 size, and CPU ms per 5 s window split into
feature extraction (stacked log-mel) and network. The distillation head is
dropped at export, so distillation is free at inference -- part of the claim.
Pass --perch-onnx to also time the Perch teacher (perch_attn_probe's cost).

Hypotheses: the ResNet trunk dominates; the GeM + attention head adds <5%
over the pooled baseline; deltas are ~free while multires triples feature
time; Perch is several times slower than the distilled student.
"""
import argparse
import time

import torch

from ..models import BirdSEDModel, MelFeatures

SR, WIN = 32000, 160000


def timed(fn, x, iters):
    with torch.inference_mode():
        for _ in range(2):
            fn(x)
        t = time.perf_counter()
        for _ in range(iters):
            out = fn(x)
    return (time.perf_counter() - t) / iters, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--batch", type=int, default=12)  # one 60 s file
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--perch-onnx", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    wav = torch.randn(args.batch, 1, WIN)

    configs = [
        ("ResNet18 + SED head, delta mel (full)", "sed", "delta", 256),
        ("ResNet18 + SED head, single mel", "sed", "single", 256),
        ("ResNet18 + SED head, multires mel", "sed", "multires", 256),
        ("ResNet18 + SED head, delta, 128 mel", "sed", "delta", 128),
        ("ResNet18 pooled (baseline), delta mel", "pooled", "delta", 256),
    ]
    print(f"{'config':42s} {'params(M)':>9s} {'MB':>6s} {'feat ms':>8s} {'net ms':>7s} "
          f"{'total ms/5s':>11s} {'x realtime':>10s}")
    for name, head, stack, mels in configs:
        feat = MelFeatures(n_mels=mels, stack_mode=stack).eval()
        model = BirdSEDModel(pretrained=False, in_chans=feat.n_channels, head=head,
                             use_distill=False).eval()
        n = sum(p.numel() for p in model.parameters())
        tf, x = timed(feat, wav, args.iters)
        tn, _ = timed(model, x, args.iters)
        f_ms, n_ms = tf / args.batch * 1000, tn / args.batch * 1000
        print(f"{name:42s} {n/1e6:9.2f} {n*4/1e6:6.1f} {f_ms:8.1f} {n_ms:7.1f} "
              f"{f_ms + n_ms:11.1f} {5000 / (f_ms + n_ms):10.0f}")

    if args.perch_onnx:
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = args.threads
        sess = ort.InferenceSession(args.perch_onnx, so, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        x = wav.squeeze(1).numpy()
        for _ in range(2):
            sess.run(None, {inp: x})
        t = time.perf_counter()
        for _ in range(args.iters):
            sess.run(None, {inp: x})
        ms = (time.perf_counter() - t) / args.iters / args.batch * 1000
        print(f"{'Perch v2 (teacher / probe backbone)':42s} {'-':>9s} {'-':>6s} {'-':>8s} {ms:7.1f} "
              f"{ms:11.1f} {5000 / ms:10.0f}")


if __name__ == "__main__":
    main()
