"""Train each ablation variant of Model_1 (5-fold CV) into its own folder.

    python -m distilled_sed.ablations.run_ablation --out runs/ablations
    python -m distilled_sed.ablations.run_ablation --only full no_distill --epochs 10

Each variant writes runs/ablations/<name>/{oof.npz, history.json, *.onnx}.
Data paths are passed through env vars (COMP_DIR, WAVEFORM_CACHE_DIR,
PSEUDO_DIR); see distilled_sed/model1_sed.py.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .configs import TRAIN_VARIANTS

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/ablations")
    ap.add_argument("--only", nargs="*", help="subset of variant names")
    ap.add_argument("--epochs", type=int, help="override EPOCHS for every variant")
    ap.add_argument("--folds", type=int, nargs="*", help="override FOLDS, e.g. 0 1")
    ap.add_argument("--debug", action="store_true", help="notebook DEBUG mode (tiny run)")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    names = args.only or list(TRAIN_VARIANTS)
    for name in names:
        out_dir = Path(args.out) / name
        if args.skip_existing and (out_dir / "oof.npz").exists():
            print(f"[skip] {name}")
            continue
        overrides = dict(TRAIN_VARIANTS[name])
        if overrides.get("script") == "perch_probe":
            cmd = [sys.executable, "-m", "distilled_sed.perch_probe",
                   "--comp-dir", os.environ["COMP_DIR"],
                   "--waveform-cache", os.environ["WAVEFORM_CACHE_DIR"],
                   "--pseudo-dir", os.environ["PSEUDO_DIR"], "--out", str(out_dir.resolve())]
            if args.folds:
                cmd += ["--folds", *map(str, args.folds)]
            print(f"\n=== {name}: Perch + attention pooling + linear probe ===", flush=True)
            subprocess.run(cmd, check=True, cwd=REPO_ROOT)
            continue
        if args.epochs:
            overrides["EPOCHS"] = args.epochs
        if args.folds:
            overrides["FOLDS"] = args.folds
        env = dict(os.environ, MODE="train", DEBUG=str(args.debug),
                   OUT_DIR=str(out_dir.resolve()), SED_OVERRIDES=json.dumps(overrides))
        print(f"\n=== {name}: {overrides} ===", flush=True)
        subprocess.run([sys.executable, "-m", "distilled_sed.model1_sed"], env=env,
                       check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
