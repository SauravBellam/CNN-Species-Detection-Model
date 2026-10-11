"""Inference cost of the ResNet SED vs simpler heads (no data needed).

    python -m distilled_sed.ablations.benchmark_cost --threads 1

Reports parameters, fp32 size, and CPU latency per 5 s window for the
inference graph only. The distillation head is dropped at export, so
distillation is free at inference time -- part of the claim being tested.
Hypothesis: the ResNet trunk dominates cost; the GeM + attention head adds
<5% over the old pooled ResNet, while fewer mel bins cut cost roughly linearly.
"""
import argparse
import time

import torch
import torch.nn as nn

from ..models import BirdSEDModel, ResNetTrunk

N_FRAMES = 32000 * 5 // 512 + 1  # 313


class PooledResNet(nn.Module):
    """The repo's old model: ResNet-18 -> global avgpool -> fc."""

    def __init__(self, n_classes=234):
        super().__init__()
        self.backbone = ResNetTrunk(pretrained=False)
        self.fc = nn.Linear(self.backbone.out_dim, n_classes)

    def forward(self, x):
        return self.fc(self.backbone(x).mean(dim=[2, 3]))


def bench(model, n_mels, batch, iters):
    model.eval()
    x = torch.randn(batch, 1, n_mels, N_FRAMES)
    with torch.inference_mode():
        for _ in range(3):
            model(x)
        t = time.perf_counter()
        for _ in range(iters):
            model(x)
    return (time.perf_counter() - t) / iters / batch * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--batch", type=int, default=12)  # one 60 s file
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    sed = lambda: BirdSEDModel(pretrained=False, use_distill=False)
    configs = [
        ("ResNet18 + SED head, 256 mel", sed, 256),
        ("ResNet18 + SED head, 128 mel", sed, 128),
        ("ResNet18 avgpool + fc (old), 256 mel", PooledResNet, 256),
        ("ResNet18 avgpool + fc (old), 128 mel", PooledResNet, 128),
    ]
    print(f"{'config':40s} {'params(M)':>9s} {'MB':>6s} {'ms/5s':>7s} {'x realtime':>10s}")
    for name, make, mels in configs:
        m = make()
        n = sum(p.numel() for p in m.parameters())
        ms = bench(m, mels, args.batch, args.iters)
        print(f"{name:40s} {n/1e6:9.2f} {n*4/1e6:6.1f} {ms:7.1f} {5000/ms:10.0f}")


if __name__ == "__main__":
    main()
