#!/usr/bin/env bash
# Full study on ONE L4 (GCP g2-standard-16: 8 cores / 16 vCPU, 64 GB RAM).
#
# Step 1 runs Perch v2 on the GPU only (perch_gpu.py) once over every 5 s window (pseudo-labels + cached
# embeddings). Step 2 trains the variants in 3 parallel lanes sharing the GPU
# (Perch is no longer run during training, so each run is light); each lane
# has 4 DataLoader workers -> 3 x (1 + 4) = 15 of 16 vCPUs. Step 3 scores
# everything, renders attention heatmaps and measures inference cost
# (the ResNet variants on CPU and GPU; Perch on GPU only).
#
# Usage (from repo root):
#   export COMP_DIR=/data/birdclef-2026
#   export WAVEFORM_CACHE_DIR=/data/waveform_cache
#   export PERCH_ONNX_PATH=/data/perch_v2.onnx        # Perch v2 .onnx OR GPU SavedModel dir; GPU-only
#   export PERCH_LABELS=/data/perch_v2_labels.csv     # Perch v2 assets/labels.csv
#   export PSEUDO_DIR=/data/pseudo
#   bash distilled_sed/scripts/run_ablations_l4.sh            # full study
#   PILOT=1 bash distilled_sed/scripts/run_ablations_l4.sh    # 2 folds x 3 epochs timing check
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${COMP_DIR:?}" "${WAVEFORM_CACHE_DIR:?}" "${PERCH_ONNX_PATH:?}" "${PERCH_LABELS:?}" "${PSEUDO_DIR:?}"

OUT=${OUT:-runs/ablations}
export CONCURRENT_RUNS=3   # lanes sharing the GPU; used to estimate GPU-hours per run
export NUM_WORKERS=${NUM_WORKERS:-4}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # stop torch/numpy oversubscribing the 16 vCPUs
# Quick-ablation budget: 3 of the 5 CV folds, EPOCHS / MAX_STEPS_PER_EPOCH from model1_sed.py
# (10 epochs x 600 batches). Pilot: 1 fold x 2 epochs, just to prove everything runs end to end.
EXTRA=(--skip-existing --folds ${FOLDS:-0 1 2})
if [[ "${PILOT:-0}" == 1 ]]; then OUT=runs/pilot; EXTRA=(--skip-existing --folds 0 --epochs 2); fi
mkdir -p "$OUT/logs"

# ---- 1. Perch pseudo-labels (once; reused by every run)
if [[ ! -f "$PSEUDO_DIR/sc_targets_fold4.npy" ]]; then
  OMP_NUM_THREADS=8 python -m distilled_sed.pseudo_label \
    --comp-dir "$COMP_DIR" --waveform-cache "$WAVEFORM_CACHE_DIR" \
    --perch-model "$PERCH_ONNX_PATH" --perch-labels "$PERCH_LABELS" --out "$PSEUDO_DIR" \
    2>&1 | tee "$OUT/logs/pseudo_label.log"
fi

# ---- 1b. precompute log-mel spectrograms of every window (once)
if [[ ! -f "$PSEUDO_DIR/sc_mel.npy" ]]; then
  python -m distilled_sed.precompute_mels --waveform-cache "$WAVEFORM_CACHE_DIR" --pseudo-dir "$PSEUDO_DIR" \
    2>&1 | tee "$OUT/logs/precompute_mels.log"
fi

# ---- 2. training lanes (baselines first so they finish early)
LANES=(
  "resnet_baseline full no_pseudo_focal no_mixup no_rare_upsample perch_attn_probe"
  "no_distill single_channel_mel mean_freq_pool no_specaug distill_no_stopgrad"
  "no_pseudo_sc mean_time_pool no_soundscape_data"
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

# ---- 3. cost (machine is idle now, so timings are clean), score, graphs
python -m distilled_sed.ablations.benchmark_cost --runs "$OUT" --perch-model "$PERCH_ONNX_PATH" \
  | tee "$OUT/cost.txt"
python -m distilled_sed.ablations.evaluate --runs "$OUT" --taxonomy "$COMP_DIR/taxonomy.csv"
python -m distilled_sed.ablations.plot --runs "$OUT"
echo "Attention heatmaps: $OUT/*/attention_heatmaps.zip"
exit $fail
