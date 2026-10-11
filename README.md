# `src` Package

This folder contains the BirdCLEF audio classification pipeline. It prepares metadata and folds, precomputes log-mel spectrogram chunks, trains a multi-label ResNet model, runs inference on clips or soundscapes, and provides debugging utilities for inspecting preprocessing and training runs.

Run commands from the repository root so imports like `src.pipeline` resolve correctly.

## Setup

Install the project dependencies:

```bash
pip install -r requirements.txt
```

The pipeline expects BirdCLEF-style data under the project base directory:

```text
train.csv
taxonomy.csv
train_audio/
train_soundscapes/
train_soundscapes_labels.csv
```

Generated artifacts are written to folders such as `splits/`, `data_processed/`, `checkpoints/`, `when_checkpoints/`, and `debug_precompute/`.

## Typical Workflow

### 1. Prepare metadata and folds

Build the master metadata tables, label maps, train/validation folds, and optional species templates:

```bash
python -m src.pipeline prepare \
  --base-dir . \
  --splits-dir splits \
  --templates-path templates.pkl \
  --test-size 0.15 \
  --n-splits 5
```

Use `--skip-templates` if you do not want to build template filters.

### 2. Precompute spectrogram chunks

Create cached `.npy` log-mel spectrogram chunks for one fold:

```bash
python -m src.pipeline precompute \
  --base-dir . \
  --splits-dir splits \
  --data-processed-dir data_processed \
  --templates-path templates.pkl \
  --fold 0 \
  --n-jobs 3 \
  --chunking-mode signal
```

Precompute every available fold:

```bash
python -m src.pipeline precompute --all-folds --base-dir .
```

Supported chunking modes are:

- `signal`: center chunks around detected signal intervals.
- `sequential`: split audio into regular fixed-length windows.
- `random`: sample random fixed-length chunks.
- `when`: use a trained WHEN detector to center chunks.
- `hybrid`: choose the chunking strategy from each row's phase group.

### 3. Train the classifier

Train one fold:

```bash
python -m src.pipeline train \
  --base-dir . \
  --splits-dir splits \
  --data-processed-dir data_processed \
  --checkpoints-dir checkpoints \
  --fold 0 \
  --epochs 10 \
  --batch-size 32 \
  --augment
```

Train all folds:

```bash
python -m src.pipeline train --all-folds --base-dir .
```

Useful training options:

- `--curriculum`: train in phase stages from clean data toward noisier data.
- `--aug-config path-or-json`: override augmentation probabilities and ranges.
- `--debug-errors`: write validation false positive, false negative, and low-confidence reports.
- `--early-stopping-patience N`: stop after `N` epochs without improvement.
- `--secondary-label-total-weight W`: downweight secondary labels while keeping the primary label at `1.0`.

### 4. Run inference

Predict the top labels for one audio file:

```bash
python -m src.pipeline infer path/to/audio.ogg \
  --checkpoint checkpoints/fold_0_best.pt \
  --top-k 10 \
  --inference-mode auto \
  --aggregation max
```

Inference modes:

- `auto`: use soundscape windows for long files or soundscape-like names; otherwise use clip chunking.
- `clip`: use signal-centered chunking for a single clip.
- `soundscape`: use sliding fixed-length soundscape windows.

Aggregation options are `max`, `mean`, and `meanmax`.

## WHEN Detector

The WHEN detector is a small auxiliary model that learns frame-level event timing from clean clips. It can be used by the precompute step with `--chunking-mode when` or `--chunking-mode hybrid`.

Train a WHEN detector:

```bash
python -m src.pipeline train-when \
  --base-dir . \
  --splits-dir splits \
  --fold 0 \
  --output-dir when_checkpoints \
  --epochs 8 \
  --batch-size 16
```

Use it during precompute:

```bash
python -m src.pipeline precompute \
  --base-dir . \
  --fold 0 \
  --chunking-mode when \
  --when-checkpoint when_checkpoints/when_fold_0_best.pt
```

## Debugging Commands

Build a small debug precompute dataset:

```bash
python -m src.pipeline debug-precompute \
  --base-dir . \
  --splits-dir splits \
  --output-dir debug_precompute \
  --fold 0 \
  --samples-per-phase 5
```

Inspect debug precompute outputs:

```bash
python -m src.pipeline debug-inspect \
  --base-dir . \
  --debug-dir debug_precompute
```

Compare training runs from a JSON spec:

```bash
python -m src.pipeline compare-runs run_specs.json \
  --output-path run_comparison.json
```

Example `run_specs.json`:

```json
[
  {
    "name": "baseline_fold_0",
    "checkpoint": "checkpoints/fold_0_best.pt",
    "debug_report": "checkpoints/fold_0_debug_report.json"
  }
]
```

## Module Map

- `pipeline.py`: command-line entry point for prepare, precompute, train, inference, and debug tasks.
- `metadata.py`: loads BirdCLEF CSVs, builds master/file-level metadata, labels, taxonomy targets, and fold artifacts.
- `splits.py`: creates holdout and cross-validation splits at the file level.
- `precompute.py`: extracts audio chunks, converts them to log-mel spectrograms, and writes fold caches.
- `audio_processing.py`: signal detection, chunking, padding/cropping, and spectrogram helpers.
- `dataset.py`: PyTorch dataset for cached spectrograms and optional audio/spec augmentation.
- `model.py`: ResNet18-based multi-label classifier for log-mel inputs.
- `train.py`: classifier training loop, curriculum learning, validation metrics, checkpoints, and error reports.
- `inference.py`: checkpoint loading and top-k prediction for clips or soundscapes.
- `when_model.py`, `when_detector.py`, `train_when.py`: auxiliary event-timing model and chunk selection helpers.
- `templates.py`: species template creation and secondary-label filtering.
- `eval_debug.py`, `debug_subset.py`, `debug_inspect.py`, `compare_runs.py`: diagnostics and run comparison utilities.
- `config.py`: shared sample rate, duration, mel, FFT, and signal chunking settings.
- `paths.py`: path resolution helpers for project-relative artifacts.
- `labels.py`: label cleaning, taxonomy labels, and target encoding helpers.

## Core Defaults

Audio and spectrogram defaults are defined in `config.py`:

```text
sample rate: 32000 Hz
chunk duration: 5.0 seconds
n_fft: 1024
hop_length: 320
n_mels: 128
frequency range: 20-14000 Hz
```

Signal-centered chunking uses the shared `CHUNK_CONFIG` thresholds from `config.py`.

## Spectrogram Math

The pipeline converts fixed-length waveform chunks into log-mel spectrograms before sending them to the neural network.

### Audio chunk length

Most training chunks are 5 seconds at 32 kHz:

```text
samples_per_chunk = sample_rate * duration
samples_per_chunk = 32000 * 5.0 = 160000 samples
```

If a chunk is shorter than 160,000 samples, it is zero-padded. If it is longer, the code crops or centers a 5-second window depending on the chunking path.

### Short-time Fourier transform

For each waveform chunk `x[n]`, the short-time Fourier transform is:

```text
X[k, t] = sum_n x[n] * w[n - tH] * exp(-j * 2*pi*k*n / N)
```

Where:

- `N` is `n_fft`.
- `H` is `hop_length`.
- `w` is the analysis window used by `librosa.stft`.
- `k` is the frequency-bin index.
- `t` is the time-frame index.

The power spectrogram is:

```text
P[k, t] = |X[k, t]|^2
```

For precomputed training chunks, the main values are:

```text
n_fft = 1024
hop_length = 320
sample_rate = 32000
frame_hop_seconds = hop_length / sample_rate
frame_hop_seconds = 320 / 32000 = 0.01 seconds
```

So each spectrogram frame advances by about 10 ms.

The approximate number of frames in a 5-second chunk is:

```text
num_frames ~= 1 + floor(samples_per_chunk / hop_length)
num_frames ~= 1 + floor(160000 / 320)
num_frames ~= 501
```

`librosa` may pad internally depending on its STFT defaults, but the important idea is that each column is a short-time frequency snapshot.

### Mel filter bank

The linear-frequency power spectrogram is projected into `n_mels` perceptual frequency bands:

```text
M[m, t] = sum_k B[m, k] * P[k, t]
```

Where:

- `M[m, t]` is the mel spectrogram.
- `B[m, k]` is the triangular mel filter-bank weight for mel band `m` and FFT bin `k`.
- `m` ranges from `0` to `n_mels - 1`.

Training uses:

```text
n_mels = 128
fmin = 20 Hz
fmax = 14000 Hz
```

Conceptually, mel frequencies follow a perceptual scale such as:

```text
mel(f) = 2595 * log10(1 + f / 700)
```

The implementation uses `librosa.feature.melspectrogram`, which builds the mel filter bank using librosa's default Slaney-style mel scaling unless explicitly changed.

### Log-power conversion

After the mel projection, the code converts power to decibels:

```text
S_db[m, t] = 10 * log10(max(M[m, t], amin) / ref)
```

In `precompute.py`, this is called as:

```python
mel_db = librosa.power_to_db(mel).astype(np.float32)
```

That uses librosa's default reference value of `ref=1.0`.

In `audio_processing.py` and the WHEN detector, some paths use:

```python
log_mel = librosa.power_to_db(mel, ref=np.max)
```

That normalizes each chunk relative to its own maximum mel power:

```text
S_db[m, t] = 10 * log10(M[m, t] / max(M))
```

### Normalization

Before model input, spectrograms are standardized:

```text
Z[m, t] = (S_db[m, t] - mean(S_db)) / (std(S_db) + 1e-6)
```

The resulting tensor shape is:

```text
[channels, n_mels, time_frames] = [1, 128, T]
```

The extra channel dimension is needed because the classifier is a ResNet-style 2D convolutional model.

### Signal-centered chunk detection

The signal chunker looks for active time regions before extracting a 5-second chunk.

It first computes an STFT power spectrogram:

```text
P[k, t] = |STFT(x)[k, t]|^2
```

Then it measures energy in a selected frequency band:

```text
E[t] = mean_k P[k, t] for fmin <= frequency[k] <= fmax
```

It converts that energy to dB:

```text
E_db[t] = 10 * log10(E[t] + epsilon)
```

Then it computes a smoothed short-term energy and a smoothed background estimate:

```text
short[t] = moving_average(E_db, energy_smooth_frames)
background[t] = moving_average(E_db, background_smooth_frames)
relative_db[t] = short[t] - background[t]
```

Frames are considered active when:

```text
relative_db[t] > threshold_db
```

Active frames are converted back to seconds with:

```text
time_seconds = frame_index * hop_length / sample_rate
```

Short events are removed, close events are merged, and chunks are centered around the remaining event intervals. If no event is found, the code can fall back to regular sequential 5-second windows.

### Model objective

The main classifier is multi-label, so each training example can contain more than one active class. The model outputs one logit per class:

```text
z_c = model(input)_c
```

The probability for class `c` is:

```text
p_c = sigmoid(z_c) = 1 / (1 + exp(-z_c))
```

Training uses binary cross entropy with logits:

```text
loss_c = -w_c * y_c * log(sigmoid(z_c)) - (1 - y_c) * log(1 - sigmoid(z_c))
```

The positive class weight is estimated from the training samples:

```text
pos_weight_c = negative_count_c / positive_count_c
```

Then it is clipped:

```text
1.0 <= pos_weight_c <= 20.0
```

Validation macro AUC is computed over classes that have at least one positive validation example.

### Inference aggregation

For an audio file with multiple chunks, the model produces a probability vector for each chunk:

```text
p_i[c] = probability for class c in chunk i
```

The final file-level probability can be aggregated three ways:

```text
max[c] = max_i p_i[c]
mean[c] = mean_i p_i[c]
meanmax[c] = 0.5 * (max_i p_i[c] + mean_i p_i[c])
```

The top-k species labels are selected from the aggregated probabilities.

### Implementation note

The cached training path in `precompute.py` uses the shared spectrogram settings from `config.py`: `n_fft=1024`, `hop_length=320`, `n_mels=128`, `fmin=20`, and `fmax=14000`.

The inference helper `audio_to_logmel` in `audio_processing.py` currently hardcodes `n_fft=1024`, `hop_length=512`, `n_mels=128`, `fmin=20`, and `fmax=8000`. If you want strict train/inference feature matching, make that helper use the constants from `config.py`.

## Checkpoint Contents

Classifier checkpoints contain:

- `class_list`: full model output classes, including taxonomy classes.
- `species_class_list`: species-only labels used for inference ranking.
- `model_state_dict`: trained PyTorch weights.
- `history`: per-epoch training and validation metrics.
- `best_val_loss`, `best_val_auc`, and `best_epoch`.
- Training options such as `dropout` and `secondary_label_total_weight`.

Use checkpoints created by `src.train` or `src.pipeline train`; older checkpoints without `class_list` and `model_state_dict` will not load in `inference.py`.

## Distilled SED (`distilled_sed/`) and ablations

`distilled_sed/model1_sed.py` is the "Model_1" distilled SED from nina2025's Kaggle notebook
[BirdCLEF 2026 EoS.9](https://www.kaggle.com/code/nina2025/birdclef-2026-eos-9#Model_1). All credit
for the model idea (GeM frequency pooling, attention SED head, Perch v2 embedding distillation with
stop-gradient) goes to the original authors. Our changes: the EfficientNet-B0 backbone is replaced by
our ResNet-18 with its avgpool/fc removed (`distilled_sed/models.py`, also used by `model.py`), and the
LR schedule is removed. The goal is a cheap model that can run 24/7 on site.

```bash
# train every ablation variant (5-fold CV each)
COMP_DIR=... WAVEFORM_CACHE_DIR=... PERCH_ONNX_PATH=... \
  python -m distilled_sed.ablations.run_ablation --out runs/ablations
# score them on many metrics and check hypotheses (writes CSV + hypotheses.md)
python -m distilled_sed.ablations.evaluate --runs runs/ablations --taxonomy $COMP_DIR/taxonomy.csv
# inference cost (params, MB, CPU ms per 5 s window)
python -m distilled_sed.ablations.benchmark_cost --threads 1
```

The metrics are macro ROC-AUC, PR-AUC, LRAP, macro F1@0.5, ECE, Brier, temporal jitter, AUC for rare
(<20 clips) and common (≥100 clips) species, and per-taxon AUC. Each hypothesis is listed in
`distilled_sed/ablations/configs.py`.

### Running on one GPU (GCP g2-standard-16, 1x L4)

```bash
bash distilled_sed/scripts/setup_vm.sh
export COMP_DIR=... WAVEFORM_CACHE_DIR=... PERCH_ONNX_PATH=...
PILOT=1 bash distilled_sed/scripts/run_ablations_l4.sh   # quick timing check
bash distilled_sed/scripts/run_ablations_l4.sh           # full study
```

This runs 3 variant lanes at the same time on the L4, each with 4 DataLoader workers.
