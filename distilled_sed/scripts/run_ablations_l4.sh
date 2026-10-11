#!/usr/bin/env bash
# Run all ablation variants on ONE L4 (g2-standard-16: 8 cores / 16 vCPU, 64 GB RAM).
#
# A single run cannot saturate the L4 (ResNet-18 is small; the bottleneck is the
# Perch teacher + data loading), so 3 lanes share the GPU. Each lane gets
# 4 DataLoader workers -> 3 x (1 main + 4 workers) = 15 of 16 vCPUs.
# Variants are balanced so each lane has similar cost (no_distill skips Perch).
#
# Usage (from repo root):
#   export COMP_DIR=/data/birdclef-2026
#   export WAVEFORM_CACHE_DIR=/data/waveform_cache
#   export PERCH_ONNX_PATH=/data/perch_v2_no_dft.onnx
#   bash distilled_sed/scripts/run_ablations_l4.sh            # full study
#   PILOT=1 bash distilled_sed/scripts/run_ablations_l4.sh    # 2 folds x 3 epochs timing check
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${COMP_DIR:?}" "${WAVEFORM_CACHE_DIR:?}" "${PERCH_ONNX_PATH:?}"

OUT=${OUT:-runs/ablations}
export NUM_WORKERS=${NUM_WORKERS:-4}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # stop torch/numpy oversubscribing the 16 vCPUs
EXTRA=(--skip-existing)
if [[ "${PILOT:-0}" == 1 ]]; then OUT=runs/pilot; EXTRA+=(--folds 0 1 --epochs 3); fi
mkdir -p "$OUT/logs"

LANES=(
  "full no_mixup no_rare_upsample"
  "distill_no_stopgrad mean_freq_pool no_specaug"
  "mean_time_pool no_soundscape_data no_distill"
)
pids=()
for i in "${!LANES[@]}"; do
  # shellcheck disable=SC2086
  python -m distilled_sed.ablations.run_ablation --out "$OUT" --only ${LANES[$i]} "${EXTRA[@]}" \
    > "$OUT/logs/lane$i.log" 2>&1 &
  pids+=($!)
  echo "lane $i (pid $!): ${LANES[$i]}  -> $OUT/logs/lane$i.log"
  sleep 60   # stagger start-up so lanes don't all load caches at once
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done

python -m distilled_sed.ablations.evaluate --runs "$OUT" --taxonomy "$COMP_DIR/taxonomy.csv"
python -m distilled_sed.ablations.benchmark_cost --threads 1 | tee "$OUT/cost.txt"
exit $fail
