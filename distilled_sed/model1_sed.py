"""Model_1 -- Distilled SED (EfficientNet-B0 student, Perch v2 teacher).

CREDIT: This file is copied from the "Model_1" section of the Kaggle notebook
"BirdCLEF 2026 EoS.9" by nina2025:
    https://www.kaggle.com/code/nina2025/birdclef-2026-eos-9#Model_1
The model idea (GeM frequency pooling + attention SED head + Perch embedding
distillation with stop-gradient) and all of the code below belong to the
original authors (the notebook itself credits tuckerarrants' distilled-SED
datasets and the BirdCLEF 2025 1st-place SED design by nikitababich).

Changes vs the notebook cell:
  * model classes moved to distilled_sed/models.py; the EfficientNet-B0 backbone
    is replaced by this repo's ResNet-18 with its avgpool/fc removed, so the
    attention head sees the full time axis;
  * the warmup + cosine LR schedule is deleted (constant LR);
  * removed the `if 'Model_1' in _ensemble_models:` wrapper, `!pip` magics
    commented out;
  * paths / MODE / DEBUG / any config constant can be overridden via env vars
    (see `_apply_overrides`) for the ablation scripts, plus three switches
    (STOP_GRAD_CLS, FREQ_POOL, TIME_POOL) that default to the original behaviour;
  * OOF predictions (clip / fmax / blend) are saved to OUT_DIR/oof.npz;
  * the inference loader now looks for the sed_distill_fold*.onnx files this
    script exports (the notebook looked for sed_fold*.onnx).
Run:  MODE=train python -m distilled_sed.model1_sed
"""
_file_name_submission = "subm_1.csv"

# Install from bundled wheel (no network needed at scoring time)
# (notebook) !pip install -q /kaggle/input/datasets/tuckerarrants/perch-v2-no-dft-onnx/onnxruntime-1.24.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl

# =================================================================
# S1 -- IMPORTS + CONFIG
# =================================================================
import os, sys, time, json, pickle, gc, random, math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast
import torchaudio
import onnxruntime as ort
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.special import expit as sigmoid_np
import warnings
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name()}")

# ------------------------------------------------------------------
# Notebook Mode
# -----------------------------------------------------------------
MODE = os.environ.get("MODE", "infer")  # "train" or "infer"  [ablation hook]

if MODE == 'train':    #Kaggle struggles with full training pipeline
    DEBUG = True     
else:
    DEBUG = False
DEBUG = os.environ.get("DEBUG", str(DEBUG)) in ("1", "True", "true")  # [ablation hook]

# ------------------------------------------------------------------
# Paths -- Kaggle dataset layout
# -----------------------------------------------------------------
COMP_DIR = Path("/kaggle/input/competitions/birdclef-2026")
WAVEFORM_CACHE_DIR = Path("/kaggle/input/datasets/tuckerarrants/birdclef-2026-waveform-cache/waveform_cache")

LABELS_PATH     = COMP_DIR / "train_soundscapes_labels.csv"
TAXONOMY_PATH   = COMP_DIR / "taxonomy.csv"
SAMPLE_SUB_PATH = COMP_DIR / "sample_submission.csv"
TEST_DIR        = COMP_DIR / "test_soundscapes"

OUT_DIR = Path("/kaggle/working")

NUM_CLASSES = 234
SR = 32000

# --- Duration ---
TRAIN_DURATION = 5    # seconds
VAL_DURATION   = 5    # always 5s for competition eval
TRAIN_SAMPLES  = SR * TRAIN_DURATION
VAL_SAMPLES    = SR * VAL_DURATION

N_FOLDS = 5

# --- Mel spectrogram ---
N_FFT      = 2048
HOP_LENGTH = 512
N_MELS     = 256
FMIN       = 20
FMAX       = 16000

# --- Model ---
BACKBONE_NAME = "resnet18"  # was tf_efficientnet_b0.ns_jft_in1k in the notebook

# --- Perch distillation ---
USE_PERCH_DISTILL = True
PERCH_EMBED_DIM   = 1536
ALPHA_DISTILL     = 1.0   # MSE loss weight

# --- Training ---
FOLDS  = [0, 1, 2, 3, 4]
EPOCHS = 25
BATCH  = 16 if DEBUG else 64        # used 64 locally, 16 for Kaggle
LR     = 5e-4
WD     = 1e-4

# --- Upsampling ---
MIN_SAMPLE = 20

# --- Augmentation ---
AUG_PROB = 0.5
AUG_GAIN_DB_RANGE      = (-6.0, 6.0)
AUG_NOISE_SNR_DB_RANGE = (10.0, 30.0)

# --- MixUp ---
USE_FOCAL_MIXUP    = True
MIXUP_PROB         = 0.5
MIXUP_ALPHA        = 0.4
MIXUP_HARD         = True    # union labels (hard) vs weighted blend (soft)

USE_FOCAL_SC_MIXUP     = True
FOCAL_SC_MIXUP_PROB    = 0.5
FOCAL_SC_MIXUP_ALPHA   = 0.4

# --- FreqMixStyle (disabled by default) ---
FREQ_MIXSTYLE_PROB  = 0.0
FREQ_MIXSTYLE_ALPHA = 0.1

# --- SpecAugment ---
FREQ_MASK_PARAM = 10
TIME_MASK_PARAM = 10
NUM_FREQ_MASKS  = 1
NUM_TIME_MASKS  = 2

# --- Source weights ---
USE_FOCAL           = True
USE_FOCAL_SECONDARY = True
USE_LABELED_SC      = True

# --- Data loading (notebook used num_workers=0; >0 keeps the GPU fed) ---
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 0))

# --- Perch pseudo labels (run distilled_sed/pseudo_label.py first) ---
PSEUDO_DIR       = Path("/kaggle/working/pseudo")
USE_PSEUDO_FOCAL = True   # Perch-localised soft targets per 5 s focal window (else clip labels)
USE_PSEUDO_SC    = True   # add unlabelled soundscapes with fused Perch + prior soft targets

# --- Input features / head ---
STACK_MODE = "delta"      # "delta" (log-mel, d, dd) | "multires" | "single" (1-ch notebook input)
HEAD       = "sed"        # "sed" (GeM + attention) | "pooled" (plain ResNet-18 baseline)

# --- Attention heatmap tracking ---
TRACK_N = 16              # fixed validation windows per fold tracked every epoch

ACTIVE_SOURCES = ["focal", "sc", "pseudo_sc"]
SHARES = {"focal": 0.7, "sc": 0.1, "pseudo_sc": 0.2}
SOURCE_WEIGHTS = {
    "focal":         1.0,
    "focal_missing": 0.0,
    "sc":            1.0,
    "pseudo_sc":     1.0,
}

# [ablation hook] env overrides --------------------------------------
def _apply_overrides():
    g = globals()
    for k in ("COMP_DIR", "WAVEFORM_CACHE_DIR", "OUT_DIR", "PSEUDO_DIR"):
        if os.environ.get(k):
            g[k] = Path(os.environ[k])
    g["LABELS_PATH"] = g["COMP_DIR"] / "train_soundscapes_labels.csv"
    g["TAXONOMY_PATH"] = g["COMP_DIR"] / "taxonomy.csv"
    g["SAMPLE_SUB_PATH"] = g["COMP_DIR"] / "sample_submission.csv"
    g["TEST_DIR"] = g["COMP_DIR"] / "test_soundscapes"
    # SED_OVERRIDES='{"USE_PERCH_DISTILL": false, "EPOCHS": 10}'
    for k, v in json.loads(os.environ.get("SED_OVERRIDES", "{}")).items():
        if k not in g:
            raise KeyError(f"unknown config key {k}")
        g[k] = tuple(v) if isinstance(g[k], tuple) else v
    g["OUT_DIR"].mkdir(parents=True, exist_ok=True)

STOP_GRAD_CLS = None   # None -> same as USE_PERCH_DISTILL (original behaviour)
FREQ_POOL = "gem"      # "gem" (original) | "mean"
TIME_POOL = "att"      # "att" (original) | "mean"
_apply_overrides()
SHARES = {k: v for k, v in SHARES.items()
          if (k != "sc" or USE_LABELED_SC) and (k != "pseudo_sc" or USE_PSEUDO_SC)}
SHARES = {k: v / sum(SHARES.values()) for k, v in SHARES.items()}

print(f"Backbone: {BACKBONE_NAME}")
print(f"Train duration: {TRAIN_DURATION}s | Mel: {N_MELS} mels, n_fft={N_FFT}, hop={HOP_LENGTH}")
print(f"Distillation: {'ON' if USE_PERCH_DISTILL else 'OFF'} (alpha={ALPHA_DISTILL}, cached Perch embeddings)")
print(f"Input: {STACK_MODE} | Head: {HEAD} | Pseudo focal: {USE_PSEUDO_FOCAL} | Pseudo sc: {USE_PSEUDO_SC}")
print(f"Batch: {BATCH} | Epochs: {EPOCHS} | Folds: {FOLDS}")


if MODE == "train":
    # (notebook) !pip install -q timm torchaudio onnxscript onnx
    pass

# =================================================================
# S2 -- LOAD DATA
# =================================================================

# --- Label ordering from sample_submission (defines column order) ---
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
LABEL2IDX = {label: idx for idx, label in enumerate(PRIMARY_LABELS)}
taxonomy = pd.read_csv(TAXONOMY_PATH, dtype={"primary_label": str})
label_to_taxon = dict(zip(taxonomy["primary_label"].astype(str),
                          taxonomy["class_name"].astype(str)))
TAXON_MASKS = {t: np.array([i for i, l in enumerate(PRIMARY_LABELS)
                            if label_to_taxon.get(l, "") == t])
               for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]}

# --- Focal recording metadata ---
audio_cache_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "audio_cache_meta.csv", dtype={"primary_label": str})
train_df = pd.read_csv(COMP_DIR / "train.csv", dtype={"primary_label": str})
audio_cache_meta = audio_cache_meta.merge(
    train_df[["filename", "secondary_labels"]], on="filename", how="left"
)
audio_cache_meta = audio_cache_meta[
    audio_cache_meta["primary_label"].isin(LABEL2IDX)
].reset_index(drop=True)
print(f"Focal audio cache: {len(audio_cache_meta)} entries")

# --- Soundscape window metadata ---
sc_cache_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "soundscape_cache_meta.csv")
sc_cache_meta["label_list"] = sc_cache_meta["label_list"].apply(
    lambda x: x.split(";") if isinstance(x, str) else []
)
print(f"Soundscape cache: {len(sc_cache_meta)} windows")

# --- Build soundscape label matrix from ground truth ---
sc_labels_raw = pd.read_csv(LABELS_PATH, dtype={"primary_label": str}).drop_duplicates()
sc_labels_raw["start_sec"] = pd.to_timedelta(sc_labels_raw["start"]).dt.total_seconds().astype(int)

Y_SC = np.zeros((len(sc_cache_meta), NUM_CLASSES), dtype=np.float32)
for i, row in sc_cache_meta.iterrows():
    matches = sc_labels_raw[
        (sc_labels_raw["filename"] == row["filename"]) &
        (sc_labels_raw["start_sec"] == row["start_sec"])
    ]
    for _, m in matches.iterrows():
        for lbl in str(m["primary_label"]).split(";"):
            lbl = lbl.strip()
            if lbl in LABEL2IDX:
                Y_SC[i, LABEL2IDX[lbl]] = 1.0

labeled_sc_mask = Y_SC.sum(axis=1) > 0
print(f"Soundscape labels: {labeled_sc_mask.sum()}/{len(Y_SC)} windows labeled, "
      f"{int(Y_SC.sum())} positives, {int((Y_SC.sum(axis=0) > 0).sum())} species")

# =================================================================
# FOLD ASSIGNMENT
# =================================================================

# --- Focal: StratifiedKFold by species ---
audio_for_split = audio_cache_meta.drop_duplicates("original_idx").reset_index(drop=True)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
audio_for_split["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(audio_for_split, audio_for_split["primary_label"])):
    audio_for_split.loc[val_idx, "fold"] = fold
audio_cache_meta = audio_cache_meta.merge(
    audio_for_split[["original_idx", "fold"]], on="original_idx", how="left"
)
print(f"\nFocal fold distribution:\n{audio_cache_meta['fold'].value_counts().sort_index()}")

# --- Soundscape: file-level folds, all 66 files distributed ---
# All files including S22 participate in CV for maximum species coverage
# (46 multi-fold species vs 32 with S22 holdout). The non_s22_mask_sc
# filter in evaluation still excludes S22 windows from the primary metric.
from sklearn.model_selection import GroupKFold

sc_files = sc_cache_meta[["filename", "site"]].drop_duplicates().reset_index(drop=True)
gkf = GroupKFold(n_splits=N_FOLDS)
sc_files["fold"] = -1
for fold, (_, val_idx) in enumerate(gkf.split(sc_files, groups=sc_files["filename"])):
    sc_files.loc[sc_files.index[val_idx], "fold"] = fold

file_to_fold = dict(zip(sc_files["filename"], sc_files["fold"]))
sc_cache_meta["fold"] = sc_cache_meta["filename"].map(file_to_fold).fillna(-1).astype(int)
print(f"\nSoundscape fold distribution:")
print(sc_cache_meta["fold"].value_counts().sort_index())
# =================================================================
# PERCH PSEUDO-LABEL CACHE: every focal clip is cut into 5 s windows; each
# window has a Perch-localised soft target and a Perch embedding (distillation
# target). Replaces the notebook's random crop + online Perch teacher.
# =================================================================
focal_win = pd.read_parquet(PSEUDO_DIR / "focal_windows.parquet")
focal_win["row"] = np.arange(len(focal_win))
FOCAL_T = np.load(PSEUDO_DIR / "focal_targets.npy", mmap_mode="r")
FOCAL_E = np.load(PSEUDO_DIR / "focal_emb.npy", mmap_mode="r")
audio_cache_meta = focal_win.merge(audio_for_split[["original_idx", "fold"]],
                                   on="original_idx", how="inner")
print(f"Focal 5 s windows: {len(audio_cache_meta)}")

sc_win = pd.read_parquet(PSEUDO_DIR / "sc_windows.parquet")
SC_E = np.load(PSEUDO_DIR / "sc_emb.npy", mmap_mode="r")
SC_EMB_ROW = {(f, int(t)): i for i, (f, t) in enumerate(zip(sc_win["filename"], sc_win["start_sec"]))}
pseudo_sc_df = sc_win[~sc_win["labeled"]].reset_index().rename(columns={"index": "row"})
print(f"Unlabelled soundscape windows (pseudo): {len(pseudo_sc_df)}")

# =================================================================
# UPSAMPLE RARE SPECIES
# =================================================================
counts = audio_cache_meta["primary_label"].value_counts()
rare_species = counts[counts < MIN_SAMPLE].index
extra_rows = []
for sp in rare_species:
    sp_rows = audio_cache_meta[audio_cache_meta["primary_label"] == sp]
    n_copies = int(np.ceil(MIN_SAMPLE / len(sp_rows))) - 1
    for _ in range(n_copies):
        extra_rows.append(sp_rows)

n_before = len(audio_cache_meta)
if extra_rows:
    audio_cache_meta = pd.concat([audio_cache_meta] + extra_rows, ignore_index=True)
print(f"\nUpsampled {len(rare_species)} rare species (min={MIN_SAMPLE}): "
      f"{n_before} -> {len(audio_cache_meta)} samples")

# Non-S22 mask for evaluation (S22 is a site with known label noise)
sc_sites = sc_cache_meta["site"].values
non_s22_mask_sc = sc_sites != "S22"
print(f"S22: {(~non_s22_mask_sc).sum()}, non-S22: {non_s22_mask_sc.sum()}")
print("OK Data loaded")


if DEBUG:
    EPOCHS = 1
    FOLDS = [0]
    audio_cache_meta = audio_cache_meta.groupby("primary_label").head(3).reset_index(drop=True)
    sc_cache_meta = sc_cache_meta.head(50)
    Y_SC = Y_SC[:50]
    non_s22_mask_sc = non_s22_mask_sc[:50]
    print(f"DEBUG MODE: {len(audio_cache_meta)} focal, {len(sc_cache_meta)} sc, "
          f"{EPOCHS} epoch, folds={FOLDS}")

# =================================================================
# S3 -- EVAL UTILITIES + MEL TRANSFORM + SED MODELS
# =================================================================

def compute_macro_auc(y_true, y_pred, mask=None, class_mask=None):
    """Macro-averaged AUC across evaluable species."""
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    if class_mask is not None:
        y_true, y_pred = y_true[:, class_mask], y_pred[:, class_mask]
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        try:
            aucs.append(roc_auc_score(col, y_pred[:, c]))
        except ValueError:
            continue
    return (np.mean(aucs) if aucs else float("nan")), len(aucs)

def full_eval(y_true, y_pred, ns22, tm):
    r = {}
    a, n = compute_macro_auc(y_true, y_pred)
    r["macro_auc_all"], r["n_all"] = round(a, 4), n
    a, n = compute_macro_auc(y_true, y_pred, mask=ns22)
    r["non_s22_macro"], r["n_ns22"] = round(a, 4), n
    for t, cm in tm.items():
        a, n = compute_macro_auc(y_true, y_pred, mask=ns22, class_mask=cm)
        r[f"non_s22_{t}"] = round(a, 4)
    return r

# Model classes live in distilled_sed/models.py (ResNet-18 trunk + Model_1 SED head)
from .models import MelFeatures, SpecAugment as _SpecAug, BirdSEDModel, SEDExportWrapper

def MelSpecTransform():
    """Stacked + standardised log-mel features (see models.MelFeatures)."""
    return MelFeatures(SR, N_FFT, HOP_LENGTH, N_MELS, FMIN, FMAX, stack_mode=STACK_MODE)

IN_CHANS = 1 if STACK_MODE == "single" else 3

def SpecAugment():
    return _SpecAug(FREQ_MASK_PARAM, TIME_MASK_PARAM, NUM_FREQ_MASKS, NUM_TIME_MASKS)


def _seed_worker(worker_id):
    # numpy RNG is not reseeded per DataLoader worker; without this every
    # worker would draw the same augmentations
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)

def make_model(pretrained=True):
    return BirdSEDModel(NUM_CLASSES, pretrained=pretrained, in_chans=IN_CHANS, head=HEAD,
                        use_distill=USE_PERCH_DISTILL,
                        embed_dim=PERCH_EMBED_DIM, stop_grad_cls=STOP_GRAD_CLS,
                        freq_pool=FREQ_POOL, time_pool=TIME_POOL).to(device)

# --- per-epoch metric tracking (curves.csv -> ablations/plot.py) ---
from .tracking import class_groups, eval_metrics, to_long, write_curves
VARIANT = OUT_DIR.name
CURVE_GROUPS = class_groups(
    PRIMARY_LABELS, label_to_taxon,
    train_df["primary_label"].astype(str).value_counts().reindex(PRIMARY_LABELS).fillna(0).values)

print("OK Model definitions ready")


# =================================================================
# S4 -- DATA PIPELINE
# =================================================================

def load_int16(path):
    """Load int16 waveform tensor to float32 in [-1, 1]."""
    waveform_int16 = torch.load(path, map_location="cpu")
    return waveform_int16.float() / 32767.0

_FC = {}
def load_focal(p):
    """Load focal waveform with simple LRU cache."""
    if p in _FC: return _FC[p]
    pp = WAVEFORM_CACHE_DIR / p
    if not pp.exists(): return None
    a = load_int16(pp).numpy()
    if len(_FC) >= 2000:
        _FC.pop(next(iter(_FC)))
    _FC[p] = a
    return a

_SC_CACHE = {}
def load_sc_waveform_from(cache_dir, cache_file):
    """Load a soundscape waveform with LRU cache."""
    key = str(cache_dir / cache_file)
    if key in _SC_CACHE: return _SC_CACHE[key]
    pp = cache_dir / cache_file
    if not pp.exists(): return None
    a = load_int16(pp).numpy()
    if len(_SC_CACHE) >= 200:
        _SC_CACHE.pop(next(iter(_SC_CACHE)))
    _SC_CACHE[key] = a
    return a

def extract_chunk_np(waveform, start_sample, n_samples):
    """Extract a chunk, left-padding if the recording is too short."""
    total = len(waveform)
    if total <= n_samples:
        return np.pad(waveform, (n_samples - total, 0))
    end = start_sample + n_samples
    if end > total:
        start_sample = max(0, total - n_samples)
    return waveform[start_sample:start_sample + n_samples]

def apply_aug(w):
    """Simple waveform augmentation: gain jitter + noise + shift."""
    if np.random.random() < AUG_PROB:
        w = w * (10 ** (np.random.uniform(*AUG_GAIN_DB_RANGE) / 20))
    if np.random.random() < AUG_PROB:
        sp = (w ** 2).mean()
        if sp > 1e-10:
            w = w + np.random.randn(*w.shape).astype(w.dtype) * np.sqrt(
                sp / (10 ** (np.random.uniform(*AUG_NOISE_SNR_DB_RANGE) / 10)))
    return w

# ------------------------------------------------------------------
# Build soundscape MixUp pool (labeled windows only)
# ------------------------------------------------------------------
sc_mixup_sources = []
_sc_file_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "soundscape_file_meta.csv")
_sc_file_dict = dict(zip(_sc_file_meta["filename"], _sc_file_meta["cache_file"]))
_labeled_rows = []
for i in range(len(sc_cache_meta)):
    row = sc_cache_meta.iloc[i]
    if Y_SC[i].sum() > 0:
        cf = _sc_file_dict.get(row["filename"])
        if cf is not None:
            _labeled_rows.append({
                "filename": row["filename"], "start_sec": int(row["start_sec"]),
                "cache_file": cf, "label_idx": i, "fold": int(row.get("fold", -1)),
            })
if _labeled_rows:
    _labeled_meta = pd.DataFrame(_labeled_rows)
    sc_mixup_sources.append((WAVEFORM_CACHE_DIR, _labeled_meta, Y_SC))
    print(f"SC MixUp pool: {len(_labeled_meta)} labeled windows")

# ------------------------------------------------------------------
# FocalDS -- with Focal-Focal AND Focal-Soundscape MixUp
# ------------------------------------------------------------------
_ZERO_EMB = np.zeros(PERCH_EMBED_DIM, np.float32)

def _sc_emb(filename, start_sec):
    i = SC_EMB_ROW.get((filename, int(start_sec)))
    return (_ZERO_EMB, 0.0) if i is None else (np.asarray(SC_E[i], np.float32), 1.0)

def _item(wav, lb, src, emb, has_emb):
    """(waveform, label, weight, mask, source, perch_emb, has_emb)"""
    return (torch.from_numpy(np.asarray(wav, np.float32)).unsqueeze(0),
            torch.from_numpy(np.asarray(lb, np.float32)),
            torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), src,
            torch.from_numpy(np.asarray(emb, np.float32)), torch.tensor(has_emb))

class FocalDS(Dataset):
    """Focal 5 s windows. Target = Perch-localised soft label (USE_PSEUDO_FOCAL)
    or the clip's hard primary + secondary labels. Focal-Focal and
    Focal-Soundscape MixUp as in the notebook; Perch embeddings are mixed with
    the same lambda for the distillation target."""
    def __init__(self, df, l2i, secondary_lookup=None,
                 sc_mixup_sources=None, fold_k=None, aug=False):
        self.df, self.l2i, self.aug = df.reset_index(drop=True), l2i, aug
        self.secondary_lookup = secondary_lookup
        self.sc_mixup_sources = sc_mixup_sources
        self.fold_k = fold_k

    def __len__(self): return len(self.df)

    def _load_chunk(self, r):
        w = load_focal(r["cache_file"])
        if w is None: return None, None, None
        ch = extract_chunk_np(w, int(r["start_sec"]) * SR, TRAIN_SAMPLES)
        if USE_PSEUDO_FOCAL:
            lb = np.asarray(FOCAL_T[int(r["row"])], np.float32)
        else:
            lb = np.zeros(NUM_CLASSES, dtype=np.float32)
            if str(r["primary_label"]) in self.l2i:
                lb[self.l2i[str(r["primary_label"])]] = 1.0
            if self.secondary_lookup is not None:
                for s in self.secondary_lookup.get(int(r["original_idx"]), []):
                    if s in self.l2i: lb[self.l2i[s]] = 1.0
        return ch, lb, np.asarray(FOCAL_E[int(r["row"])], np.float32)

    def __getitem__(self, i):
        ch1, lb1, e1 = self._load_chunk(self.df.iloc[i])
        if ch1 is None:
            return _item(np.zeros(TRAIN_SAMPLES), np.zeros(NUM_CLASSES), "focal_missing", _ZERO_EMB, 0.0)

        # Focal-Focal MixUp
        if USE_FOCAL_MIXUP and self.aug and np.random.random() < MIXUP_PROB:
            ch2 = None
            for _ in range(3):
                ch2, lb2, e2 = self._load_chunk(self.df.iloc[np.random.randint(len(self.df))])
                if ch2 is not None: break
            if ch2 is not None:
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                ch_mix = apply_aug((lam * ch1 + (1 - lam) * ch2).astype(np.float32))
                lb = np.maximum(lb1, lb2) if MIXUP_HARD else (lam * lb1 + (1 - lam) * lb2)
                return _item(ch_mix, lb, "focal", lam * e1 + (1 - lam) * e2, 1.0)

        # Focal-Soundscape MixUp
        if (USE_FOCAL_SC_MIXUP and self.aug and self.sc_mixup_sources
                and np.random.random() < FOCAL_SC_MIXUP_PROB):
            cache_dir, meta_df_sc, labels = self.sc_mixup_sources[np.random.randint(len(self.sc_mixup_sources))]
            eligible = meta_df_sc[meta_df_sc["fold"] != self.fold_k] if self.fold_k is not None else meta_df_sc
            if len(eligible) > 0:
                sc_row = eligible.iloc[np.random.randint(len(eligible))]
                sc_wav = load_sc_waveform_from(cache_dir, sc_row["cache_file"])
                if sc_wav is not None and len(sc_wav) >= TRAIN_SAMPLES:
                    sc_chunk = extract_chunk_np(sc_wav, int(sc_row["start_sec"]) * SR, TRAIN_SAMPLES)
                    lam = np.random.beta(FOCAL_SC_MIXUP_ALPHA, FOCAL_SC_MIXUP_ALPHA)
                    ch_mix = apply_aug((lam * ch1 + (1 - lam) * sc_chunk).astype(np.float32))
                    lb_sc = labels[int(sc_row["label_idx"])].astype(np.float32)
                    lb = np.maximum(lb1, lb_sc) if MIXUP_HARD else lam * lb1 + (1 - lam) * lb_sc
                    e_sc, has = _sc_emb(sc_row["filename"], sc_row["start_sec"])
                    emb = lam * e1 + (1 - lam) * e_sc if has else e1
                    return _item(ch_mix, lb, "focal", emb, 1.0)

        # No MixUp
        if self.aug: ch1 = apply_aug(ch1)
        return _item(ch1, lb1, "focal", e1, 1.0)

# ------------------------------------------------------------------
# ScDS -- Labeled soundscape windows
# ------------------------------------------------------------------
class ScDS(Dataset):
    def __init__(self, Y, sc_df, aug=False):
        self.Y, self.df, self.aug = Y, sc_df.reset_index(drop=True), aug
    def __len__(self): return len(self.Y)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav_full = load_sc_waveform_from(WAVEFORM_CACHE_DIR, row.get("cache_file")) \
                   if row.get("cache_file") else None
        if wav_full is None:
            chunk = np.zeros(TRAIN_SAMPLES, np.float32)
        else:
            chunk = extract_chunk_np(wav_full, int(row["start_sec"]) * SR, TRAIN_SAMPLES)
            if self.aug: chunk = apply_aug(chunk)
        emb, has = _sc_emb(row["filename"], row["start_sec"])
        return _item(chunk, self.Y[i], "sc", emb, has)

# ------------------------------------------------------------------
# PseudoScDS -- Unlabelled soundscape windows with Perch + prior soft targets
# ------------------------------------------------------------------
class PseudoScDS(Dataset):
    def __init__(self, df, targets, aug=False):
        self.df, self.T, self.aug = df.reset_index(drop=True), targets, aug
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        w = np.load(PSEUDO_DIR / "sc_wave" / f"{Path(r['filename']).stem}.npy", mmap_mode="r")
        s0 = int(r["start_sec"]) * SR
        chunk = np.asarray(w[s0:s0 + TRAIN_SAMPLES], np.float32) / 32767.0
        if self.aug: chunk = apply_aug(chunk)
        return _item(chunk, self.T[int(r["row"])], "pseudo_sc", SC_E[int(r["row"])], 1.0)

# ------------------------------------------------------------------
# Load focal secondary labels
# ------------------------------------------------------------------
focal_secondary_labels = None
if USE_FOCAL_SECONDARY:
    focal_secondary_labels = {}
    for idx, row in train_df.iterrows():
        sec = row.get("secondary_labels", "")
        if pd.isna(sec) or sec in ("", "[]"): continue
        try:
            sec_list = eval(sec) if isinstance(sec, str) else []
        except: continue
        valid = [s for s in sec_list if s in LABEL2IDX]
        if valid: focal_secondary_labels[idx] = valid
    print(f"Focal secondary labels: {len(focal_secondary_labels)} files")

# ------------------------------------------------------------------
# Multi-source batch sampler
# ------------------------------------------------------------------
class MixSamp(torch.utils.data.Sampler):
    """Controls batch composition via per-source shares."""
    def __init__(self, sizes, names, shares, bs, nst, seed=0):
        self.sizes, self.names, self.bs, self.nst = sizes, names, bs, nst
        self.rng = np.random.default_rng(seed)
        per_src = [max(1, int(round(bs * shares.get(n, 0.0)))) for n in names]
        total = sum(per_src)
        if total != bs:
            per_src[int(np.argmax(per_src))] += (bs - total)
        self.per_src = per_src
        self.offsets = [0]
        for s in sizes[:-1]:
            self.offsets.append(self.offsets[-1] + s)
    def __len__(self): return self.nst
    def __iter__(self):
        for _ in range(self.nst):
            batch = []
            for off, size, n in zip(self.offsets, self.sizes, self.per_src):
                if n <= 0 or size <= 0: continue
                idxs = self.rng.integers(0, size, size=n)
                batch.extend([off + int(i) for i in idxs])
            self.rng.shuffle(batch)
            yield batch

def collate_m(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]),
            torch.stack([b[3] for b in batch]),
            [b[4] for b in batch],
            torch.stack([b[5] for b in batch]),
            torch.stack([b[6] for b in batch]))

def mk_sw(sr):
    """Per-sample source weight tensor."""
    return torch.tensor([SOURCE_WEIGHTS.get(s, 0.0) for s in sr], dtype=torch.float32)

print("OK Data pipeline ready")


# =================================================================
# S5 -- TRAINING
# =================================================================

def _load_val_waveforms(val_sc_df):
    """Load validation waveforms (always 5s)."""
    sc_file_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "soundscape_file_meta.csv")
    sc_file_dict = dict(zip(sc_file_meta["filename"], sc_file_meta["cache_file"]))
    wavs = []
    for _, row in val_sc_df.iterrows():
        cf = sc_file_dict.get(row["filename"])
        if cf is not None:
            w = load_sc_waveform_from(WAVEFORM_CACHE_DIR, cf)
            if w is not None:
                chunk = extract_chunk_np(w, int(row["start_sec"]) * SR, VAL_SAMPLES)
                wavs.append(torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0))
            else: wavs.append(torch.zeros(1, VAL_SAMPLES))
        else: wavs.append(torch.zeros(1, VAL_SAMPLES))
    return wavs

def _predict_from_waveforms(model, mel_transform, wav_list, batch_size=64):
    """Inference: mel -> model -> sigmoid. Distillation head is NOT used."""
    model.eval()
    preds_clip, preds_fmax, preds_blend = [], [], []
    with torch.no_grad():
        for s in range(0, len(wav_list), batch_size):
            batch = torch.stack(wav_list[s:s+batch_size]).to(device)
            mel = mel_transform(batch)
            with autocast():
                clip_logits, framewise = model(mel, return_framewise=True)
                frame_max = framewise.max(dim=1).values
                p_clip = torch.sigmoid(clip_logits).cpu().numpy()
                p_fmax = torch.sigmoid(frame_max).cpu().numpy()
                p_blend = 0.5 * p_clip + 0.5 * p_fmax
            preds_clip.append(p_clip); preds_fmax.append(p_fmax); preds_blend.append(p_blend)
    return {"clip": np.concatenate(preds_clip), "fmax": np.concatenate(preds_fmax),
            "blend": np.concatenate(preds_blend)}

def build_active_datasets(fold_k):
    items = []
    if USE_FOCAL:
        fds = FocalDS(audio_cache_meta[audio_cache_meta["fold"] != fold_k],
                      LABEL2IDX, secondary_lookup=focal_secondary_labels,
                      sc_mixup_sources=sc_mixup_sources if USE_FOCAL_SC_MIXUP else None,
                      fold_k=fold_k, aug=True)
        items.append(("focal", fds, len(fds)))
    if USE_LABELED_SC:
        vm = sc_cache_meta["fold"].values == fold_k
        sc_train_df = sc_cache_meta[~vm].reset_index(drop=True)
        Y_tr = Y_SC[~vm]
        sds = ScDS(Y_tr, sc_train_df, aug=True)
        items.append(("sc", sds, len(sds)))
    if USE_PSEUDO_SC and len(pseudo_sc_df):
        T_k = np.load(PSEUDO_DIR / f"sc_targets_fold{fold_k}.npy", mmap_mode="r")  # priors exclude fold k
        pds = PseudoScDS(pseudo_sc_df, T_k, aug=True)
        items.append(("pseudo_sc", pds, len(pds)))
    return items

def train_fold(fold_k):
    vm = sc_cache_meta["fold"].values == fold_k
    Y_val = Y_SC[vm]
    ns22_val = non_s22_mask_sc[vm]
    val_sc_df = sc_cache_meta[vm].reset_index(drop=True)

    active = build_active_datasets(fold_k)
    names, datasets, sizes = zip(*active)
    mds = ConcatDataset(list(datasets))
    nst = max(100, int(sum(sizes) / BATCH))

    print(f"  Streams: {dict(zip(names, sizes))}  steps/ep: {nst}")

    m = make_model()
    mel_transform = MelSpecTransform().to(device)
    spec_augment = SpecAugment().to(device)

    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    scaler = GradScaler()

    history = {"ep": [], "train_loss": [], "cls_loss": [], "dist_loss": [],
               "macro": [], "ns22_macro": [],
               "ns22_Aves": [], "ns22_Amphibia": [], "ns22_Insecta": [], "ns22_Mammalia": [],
               "val_preds": [], "epoch_metrics": []}
    best_ns22, best_state_ns22 = -1.0, None
    best_macro, best_state_macro = -1.0, None
    val_wavs = _load_val_waveforms(val_sc_df)

    # --- attention heatmap tracking: fixed labelled non-S22 val windows ---
    _cand = np.where((Y_val.sum(1) > 0) & ns22_val)[0][:TRACK_N]
    track = {"idx": _cand, "cls": Y_val[_cand].argmax(1) if len(_cand) else np.zeros(0, int),
             "row_id": [f"{val_sc_df.loc[i, 'filename']}_{int(val_sc_df.loc[i, 'start_sec']) + 5}" for i in _cand],
             "time_att": [], "cam": [], "prob": []}
    if len(_cand):
        with torch.no_grad():
            track["mel"] = mel_transform(torch.stack([val_wavs[i] for i in _cand]).to(device))[:, 0].cpu().numpy()

    for ep in range(EPOCHS):
        m.train()
        smp = MixSamp(list(sizes), list(names), SHARES, BATCH, nst, seed=42 + ep)
        tl = DataLoader(mds, batch_sampler=smp, collate_fn=collate_m,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        worker_init_fn=_seed_worker, persistent_workers=False)
        el, el_cls, el_dist, nb_count = 0.0, 0.0, 0.0, 0
        t0 = time.time()

        for wav, lb, wt, mk, sr, perch_emb, has_emb in tl:
            wav, lb, wt, mk = wav.to(device), lb.to(device), wt.to(device), mk.to(device)
            perch_emb, has_emb = perch_emb.to(device), has_emb.to(device)
            sw = mk_sw(sr).to(device)

            with torch.no_grad():
                mel = mel_transform(wav)
                mel = spec_augment(mel)

            with autocast():
                if USE_PERCH_DISTILL:
                    clip_logits, framewise, distill_emb = m(mel, return_framewise=True,
                                                            return_distill=True)
                else:
                    clip_logits, framewise = m(mel, return_framewise=True)

                frame_max_logits = framewise.max(dim=1).values

                # Classification loss
                bce_clip = F.binary_cross_entropy_with_logits(clip_logits, lb, reduction="none")
                bce_frame = F.binary_cross_entropy_with_logits(frame_max_logits, lb, reduction="none")
                bce = 0.5 * bce_clip + 0.5 * bce_frame
                ps = (bce * wt * mk).sum(1) / (mk.sum(1) + 1e-8)
                cls_loss = (ps * sw).mean()

                # Distillation loss
                # Distillation loss (cached Perch embedding of the same 5 s window)
                if USE_PERCH_DISTILL:
                    per = ((distill_emb.float() - perch_emb) ** 2).mean(dim=1)
                    distill_loss = (per * has_emb).sum() / has_emb.sum().clamp(min=1.0)
                    loss = cls_loss + ALPHA_DISTILL * distill_loss
                else:
                    distill_loss = torch.tensor(0.0)
                    loss = cls_loss

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            el += loss.item(); el_cls += cls_loss.item()
            el_dist += distill_loss.item(); nb_count += 1

        # Validation
        val_preds_dict = _predict_from_waveforms(m, mel_transform, val_wavs)
        val_preds = val_preds_dict["blend"]
        r = full_eval(Y_val, val_preds, ns22_val, TAXON_MASKS)
        for mode in ["clip", "fmax", "blend"]:
            r_mode = full_eval(Y_val, val_preds_dict[mode], ns22_val, TAXON_MASKS)
            r[f"ns22_{mode}"] = r_mode["non_s22_macro"]

        history["ep"].append(ep)
        history["train_loss"].append(round(el / nb_count, 5))
        history["cls_loss"].append(round(el_cls / nb_count, 5))
        history["dist_loss"].append(round(el_dist / nb_count, 5))
        history["macro"].append(r["macro_auc_all"])
        history["ns22_macro"].append(r["non_s22_macro"])
        for t in ["Aves", "Amphibia", "Insecta", "Mammalia"]:
            history[f"ns22_{t}"].append(r[f"non_s22_{t}"])
        history["val_preds"].append(val_preds.astype(np.float32))
        em = eval_metrics(Y_val, val_preds, val_sc_df["filename"].values,
                          val_sc_df["start_sec"].values, ns22_val, CURVE_GROUPS)
        em.update(train_loss=el / nb_count, cls_loss=el_cls / nb_count, dist_loss=el_dist / nb_count,
                  auc_head_clip=r["ns22_clip"], auc_head_fmax=r["ns22_fmax"],
                  epoch_seconds=time.time() - t0)
        history["epoch_metrics"].append(em)
        if len(track["idx"]):
            m.eval()
            with torch.no_grad():
                am = m.attention_maps(mel_transform(torch.stack([val_wavs[i] for i in track["idx"]]).to(device)))
            ar = np.arange(len(track["idx"]))
            track["time_att"].append(am["time_att"].float().cpu().numpy()[ar, track["cls"]])
            track["cam"].append(am["cam"].float().cpu().numpy()[ar, track["cls"]])
            track["prob"].append(am["clip_prob"].float().cpu().numpy()[ar, track["cls"]])

        tag_ns22 = ""; tag_macro = ""
        if r["non_s22_macro"] > best_ns22:
            best_ns22 = r["non_s22_macro"]
            best_state_ns22 = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            tag_ns22 = " *ns22"
        if r["macro_auc_all"] > best_macro:
            best_macro = r["macro_auc_all"]
            best_state_macro = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            tag_macro = " *macro"

        dist_str = f" dist={el_dist/nb_count:.4f}" if USE_PERCH_DISTILL else ""
        print(f"    Ep{ep:02d}: loss={el/nb_count:.4f} cls={el_cls/nb_count:.4f}{dist_str} "
              f"lr={opt.param_groups[0]['lr']:.1e} | "
              f"ns22: {r['ns22_blend']:.4f} | "
              f"Av={r['non_s22_Aves']:.4f} Am={r['non_s22_Amphibia']:.4f} "
              f"In={r['non_s22_Insecta']:.4f} Ma={r['non_s22_Mammalia']:.4f} "
              f"[{time.time()-t0:.0f}s]{tag_ns22}{tag_macro}")

    history["track"] = track
    del m, mel_transform, spec_augment
    torch.cuda.empty_cache(); gc.collect()
    return best_state_ns22, best_state_macro, history

print("OK Training function ready")


# =================================================================
# S6 -- FOLD LOOP + ONNX EXPORT
# =================================================================

if MODE != "train":
    print("Skipping training (MODE='infer')")
    oof_ns22 = None
    all_hist = {}
else:

    oof_ns22 = np.full((len(sc_cache_meta), NUM_CLASSES), np.nan, dtype=np.float32)
    oof_modes = {k: oof_ns22.copy() for k in ("clip", "fmax", "blend")}  # [ablation hook]
    all_hist = {}
    # --- compute tracking: architecture config (for benchmark_cost.py) + training cost ---
    with open(OUT_DIR / "config.json", "w") as _f:
        json.dump({"kind": "sed", "HEAD": HEAD, "STACK_MODE": STACK_MODE, "FREQ_POOL": FREQ_POOL,
                   "TIME_POOL": TIME_POOL, "N_MELS": N_MELS, "N_FFT": N_FFT,
                   "HOP_LENGTH": HOP_LENGTH, "NUM_CLASSES": NUM_CLASSES,
                   "USE_PERCH_DISTILL": USE_PERCH_DISTILL, "EPOCHS": EPOCHS, "BATCH": BATCH,
                   "FOLDS": list(FOLDS)}, _f, indent=1)
    _train_t0, _fold_secs = time.time(), {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for fold_k in FOLDS:
        print(f"\n{'='*60}")
        print(f"FOLD {fold_k}")
        print(f"{'='*60}")
        vm = sc_cache_meta["fold"].values == fold_k
        val_sc_df_k = sc_cache_meta[vm].reset_index(drop=True)
    
        _tf = time.time()
        best_ns22_state, best_macro_state, hist = train_fold(fold_k)
        _fold_secs[fold_k] = time.time() - _tf
        all_hist[fold_k] = hist

        # attention heatmap tracking for this fold -> OUT_DIR/attention/fold{k}.npz
        _tr = hist["track"]
        if len(_tr["idx"]):
            (OUT_DIR / "attention").mkdir(exist_ok=True)
            np.savez_compressed(
                OUT_DIR / "attention" / f"fold{fold_k}.npz",
                mel=_tr["mel"], time_att=np.stack(_tr["time_att"]), cam=np.stack(_tr["cam"]),
                prob=np.stack(_tr["prob"]), row_id=np.array(_tr["row_id"]),
                species=np.array([PRIMARY_LABELS[c] for c in _tr["cls"]]),
                ns22_auc=np.array(hist["ns22_macro"]))
    
        mel_tf = MelSpecTransform().to(device)
        val_wavs_k = _load_val_waveforms(val_sc_df_k)
    
        if best_macro_state is not None:
            # Save PyTorch checkpoint
            torch.save(best_macro_state, OUT_DIR / f"fold{fold_k}_best_ns22.pt")
            m = make_model()
            m.load_state_dict(best_macro_state, strict=False)
            _pd = _predict_from_waveforms(m, mel_tf, val_wavs_k)
            oof_ns22[vm] = _pd["blend"]
            for _k in oof_modes: oof_modes[_k][vm] = _pd[_k]  # [ablation hook]
    
            # --- ONNX Export (Conv1d remap for stable tracing) ---
            m.eval()
            INF_N_MELS = N_MELS
            INF_N_FRAMES = VAL_SAMPLES // HOP_LENGTH + 1
    
            export_model = SEDExportWrapper(m).to(device)
            export_model.eval()
    
            dummy_mel = torch.randn(1, IN_CHANS, INF_N_MELS, INF_N_FRAMES).to(device)
            onnx_path = OUT_DIR / f"sed_distill_fold{fold_k}.onnx"
            torch.onnx.export(
                export_model, dummy_mel, str(onnx_path),
                input_names=["mel"],
                output_names=["clip_logits", "framewise_logits"],
                dynamic_axes={"mel": {0: "batch"},
                              "clip_logits": {0: "batch"},
                              "framewise_logits": {0: "batch"}},
                opset_version=17,
            )
    
            # Verify
            _sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
            _onnx_out = _sess.run(None, {'mel': dummy_mel.cpu().numpy()})
            with torch.no_grad():
                _ref_clip, _ref_frame = export_model(dummy_mel)
            _diff = np.abs(_ref_clip.cpu().numpy() - _onnx_out[0]).max()
            print(f"  ONNX verify: max|diff|={_diff:.3e}")
            assert _diff < 1e-3, f"ONNX export diverged: {_diff}"
            del _sess
    
            size_mb = onnx_path.stat().st_size / 1e6
            print(f"  Exported {onnx_path.name} ({size_mb:.1f} MB)")
            del m, export_model

# =================================================================
# S7 -- EVALUATION SUMMARY
# =================================================================

if MODE != "train":
    print("Skipping evaluation (MODE='infer')")
else:

    has = ~np.isnan(oof_ns22[:, 0])
    # training compute. Runs share the GPU (CONCURRENT_RUNS lanes), so the GPU-hours
    # estimate divides wall time by the number of lanes running at once.
    _wall = time.time() - _train_t0
    _lanes = int(os.environ.get("CONCURRENT_RUNS", 1))
    _ep_secs = [em["epoch_seconds"] for h in all_hist.values() for em in h["epoch_metrics"]]
    with open(OUT_DIR / "train_cost.json", "w") as _f:
        json.dump({
            "wall_seconds": _wall, "fold_seconds": {str(k): v for k, v in _fold_secs.items()},
            "mean_epoch_seconds": float(np.mean(_ep_secs)) if _ep_secs else None,
            "epochs": EPOCHS, "folds": len(FOLDS), "batch": BATCH,
            "concurrent_runs": _lanes, "gpu_hours_est": _wall / 3600 / _lanes,
            "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        }, _f, indent=1)
    # [ablation hook] save everything the ablation evaluator needs
    np.savez_compressed(
        OUT_DIR / "oof.npz", y=Y_SC, ns22=non_s22_mask_sc,
        filename=sc_cache_meta["filename"].values.astype(str),
        start_sec=sc_cache_meta["start_sec"].values.astype(int),
        fold=sc_cache_meta["fold"].values.astype(int),
        labels=np.array(PRIMARY_LABELS), **oof_modes,
        train_counts=train_df["primary_label"].astype(str).value_counts()
                     .reindex(PRIMARY_LABELS).fillna(0).astype(int).values,
    )
    # render attention heatmaps (PNG per tracked window + epoch evolution) and zip them
    from .attention_viz import render_dir
    if (OUT_DIR / "attention").exists():
        render_dir(OUT_DIR / "attention")
    with open(OUT_DIR / "history.json", "w") as _f:
        json.dump({str(k): {kk: vv for kk, vv in h.items() if kk not in ("val_preds", "track")}
                   for k, h in all_hist.items()}, _f)
    if has.sum() > 0:
        r_all = full_eval(Y_SC[has], oof_ns22[has], non_s22_mask_sc[has], TAXON_MASKS)
        print("=" * 60)
        print("OOF RESULTS (best-ns22 checkpoints)")
        print("=" * 60)
        print(f"  macro AUC (all):        {r_all['macro_auc_all']:.4f}")
        print(f"  macro AUC (non-S22):    {r_all['non_s22_macro']:.4f}")
        for t in ["Aves", "Amphibia", "Insecta", "Mammalia"]:
            print(f"    {t:<12}: {r_all.get(f'non_s22_{t}', float('nan')):.4f}")
    
    # Per-epoch curves for every metric: per fold and pooled over folds -> curves.csv
    _rows = []
    for k, h in all_hist.items():
        for ep, em in enumerate(h["epoch_metrics"]):
            _rows += to_long(VARIANT, f"fold{k}", ep, em)
    _ks = sorted(all_hist)
    for ep in range(min(len(all_hist[k]["val_preds"]) for k in _ks) if _ks else 0):
        _vm = [sc_cache_meta["fold"].values == k for k in _ks]
        pooled = eval_metrics(
            np.concatenate([Y_SC[v] for v in _vm]),
            np.concatenate([all_hist[k]["val_preds"][ep] for k in _ks]),
            np.concatenate([sc_cache_meta["filename"].values[v] for v in _vm]),
            np.concatenate([sc_cache_meta["start_sec"].values[v] for v in _vm]),
            np.concatenate([non_s22_mask_sc[v] for v in _vm]), CURVE_GROUPS)
        for key in ("train_loss", "cls_loss", "dist_loss", "epoch_seconds"):
            pooled[key] = float(np.mean([all_hist[k]["epoch_metrics"][ep][key] for k in _ks]))
        _rows += to_long(VARIANT, "pooled", ep, pooled)
    write_curves(OUT_DIR / "curves.csv", _rows)

    # Per-epoch progression
    print("\nPer-epoch pooled non-S22 AUC:")
    fold_true, fold_ns22_m = {}, {}
    for fk in range(N_FOLDS):
        vm = sc_cache_meta["fold"].values == fk
        fold_true[fk] = Y_SC[vm]
        fold_ns22_m[fk] = non_s22_mask_sc[vm]
    
    n_eps = [len(all_hist[k]["val_preds"]) for k in range(N_FOLDS) if k in all_hist]
    max_ep = min(n_eps) if n_eps else 0
    for ep in range(max_ep):
        pp = np.concatenate([all_hist[k]["val_preds"][ep] for k in range(N_FOLDS) if k in all_hist])
        pt = np.concatenate([fold_true[k] for k in range(N_FOLDS) if k in all_hist])
        pm = np.concatenate([fold_ns22_m[k] for k in range(N_FOLDS) if k in all_hist])
        ns, _ = compute_macro_auc(pt, pp, mask=pm)
        print(f"  Ep{ep:02d}: {ns:.4f}")


# =================================================================
# INFERENCE -- Mel features (same torch module as training)
# =================================================================
if MODE != "infer":
    print("Skipping inference setup (MODE='train')")
else:
    import librosa

    INF_N_MELS   = 256
    INF_N_FFT    = 2048
    INF_HOP      = 512
    INF_FMIN     = 20
    INF_FMAX     = 16000
    INF_TOP_DB   = 80
    INF_SR       = 32000
    INF_CHUNK_S  = 5
    INF_CHUNK_N  = INF_SR * INF_CHUNK_S   # 160,000
    INF_N_FRAMES = INF_CHUNK_N // INF_HOP + 1  # 313
    
    _inf_features = MelSpecTransform().eval()  # identical to training features (CPU)

    def audio_to_mel(chunks):
        """Raw audio chunks (N, 160000) -> stacked standardised log-mel (N, C, 256, 313)."""
        with torch.no_grad():
            return _inf_features(torch.from_numpy(chunks)).numpy().astype(np.float32)
    
    print(f"Inference mel: {INF_N_MELS} mels, {INF_N_FRAMES} frames/chunk")


# =================================================================
# INFERENCE -- Load ONNX sessions
# =================================================================
if MODE != "infer":
    print("Skipping ONNX loading (MODE='train')")
else:
    import re, glob

    def discover_folds(sed_dir):
        pat = re.compile(r'sed_distill_fold(\d+)\.onnx$')
        folds = []
        for fname in os.listdir(sed_dir):
            m = pat.match(fname)
            if m: folds.append(int(m.group(1)))
        return sorted(folds)
    
    def make_session(onnx_path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    
    SED_DIR_CANDIDATES = [
        str(OUT_DIR),
        '/kaggle/input/datasets/tuckerarrants/bc2026-distilled-sed-public',
    ]
    SED_DIR = next((p for p in SED_DIR_CANDIDATES if os.path.isdir(p) and
                    any(f.endswith('.onnx') and 'sed' in f for f in os.listdir(p))), None)
    assert SED_DIR, f'No SED ONNX files found in {SED_DIR_CANDIDATES}'
    
    INF_FOLDS = discover_folds(SED_DIR)
    assert INF_FOLDS, f'No sed_distill_fold*.onnx in {SED_DIR}'
    print(f'Found {len(INF_FOLDS)} fold(s) in {SED_DIR}: {INF_FOLDS}')
    
    fold_sessions = []
    for fold in INF_FOLDS:
        p = f'{SED_DIR}/sed_distill_fold{fold}.onnx'
        sess = make_session(p)
        fold_sessions.append(sess)
        size_mb = os.path.getsize(p) / 1e6
        print(f'  fold {fold}: {size_mb:5.1f}MB  providers={sess.get_providers()}')


# =================================================================
# INFERENCE -- Audio loading + main loop
# =================================================================
from scipy.ndimage import convolve1d

GAUSSIAN_KERNEL = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
N_WINDOWS = 12  # 60s / 5s

if MODE != "infer":
    print("Skipping inference (MODE='train')")
else:
    try:
        import soundfile as sf
        DECODER = 'soundfile'
    except ImportError:
        DECODER = 'librosa'
    print(f'Audio decoder: {DECODER}')

    def load_audio_32k_mono(path):
        if DECODER == 'soundfile':
            wav, sr = sf.read(path, dtype='float32', always_2d=False)
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != INF_SR:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=INF_SR)
            return wav.astype(np.float32)
        else:
            wav, _ = librosa.load(path, sr=INF_SR, mono=True)
            return wav.astype(np.float32)

    def file_to_chunks(path):
        wav = load_audio_32k_mono(path)
        target_len = 60 * INF_SR
        if len(wav) < target_len:
            wav = np.pad(wav, (0, target_len - len(wav)))
        elif len(wav) > target_len:
            wav = wav[:target_len]
        n_chunks = target_len // INF_CHUNK_N
        chunks = wav[:n_chunks * INF_CHUNK_N].reshape(n_chunks, INF_CHUNK_N)
        end_times = np.arange(1, n_chunks + 1) * INF_CHUNK_S
        return chunks.astype(np.float32), end_times

    def sigmoid_inf(x):
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))),
            np.exp(np.clip(x, -50, 50)) / (1.0 + np.exp(np.clip(x, -50, 50))),
        ).astype(np.float32)

    def gauss_smooth_final(scores, weights=GAUSSIAN_KERNEL):
        """Gaussian smooth predictions across the 12 windows within each file.
        Operates on the leading axis of the per-file slab."""
        smoothed = scores.reshape(-1, N_WINDOWS, scores.shape[1]).copy()
        for i in range(smoothed.shape[0]):
            smoothed[i] = convolve1d(smoothed[i], weights, axis=0, mode='nearest')
        return smoothed.reshape(-1, scores.shape[1])

    # Discover test files (with train_soundscapes fallback for debugging)
    test_files = sorted(glob.glob(f'{TEST_DIR}/*.ogg')) if TEST_DIR.is_dir() else []
    if len(test_files) == 0:
        fallback = COMP_DIR / "train_soundscapes"
        if fallback.is_dir():
            test_files = sorted(glob.glob(f'{fallback}/*.ogg'))[:5]
            print(f'No test_soundscapes -- using {len(test_files)} train files for debug')
    print(f'Test files: {len(test_files)}')

    # --- Main inference loop ---
    t0 = time.time()
    all_rows, all_preds = [], []

    for file_idx, file_path in enumerate(test_files):
        basename = os.path.basename(file_path).replace('.ogg', '')
        chunks, end_times = file_to_chunks(file_path)
        mel = audio_to_mel(chunks)

        # Accumulate logits in logit space across folds (matches the
        # smoothing-in-logit-space convention from the other inference notebook).
        # Per fold: blend clip + frame_max in LOGIT space.
        logits_sum = np.zeros((chunks.shape[0], NUM_CLASSES), dtype=np.float32)
        for sess in fold_sessions:
            outs = sess.run(None, {'mel': mel})
            clip_logits = outs[0]
            frame_max = outs[1].max(axis=1)
            logits_sum += 0.5 * clip_logits + 0.5 * frame_max
        logits_mean = logits_sum / len(INF_FOLDS)

        # Smooth across windows in logit space, then sigmoid once.
        logits_smoothed = gauss_smooth_final(logits_mean)
        probs = sigmoid_inf(logits_smoothed)

        all_rows.extend([f'{basename}_{int(t)}' for t in end_times])
        all_preds.append(probs)

        if (file_idx + 1) % 50 == 0 or file_idx == 0 or file_idx == len(test_files) - 1:
            elapsed = time.time() - t0
            rate = (file_idx + 1) / elapsed
            print(f'  [{file_idx+1:4d}/{len(test_files)}] {elapsed:.1f}s  {rate:.2f} files/s')

    all_preds_arr = np.concatenate(all_preds) if all_preds else np.zeros((0, NUM_CLASSES), np.float32)
    print(f'\nInference: {len(all_rows)} rows, {time.time()-t0:.1f}s total')

# =================================================================
# WRITE SUBMISSION
# =================================================================
if MODE != "infer":
    print("Skipping submission (MODE='train')")
else:
    submission = pd.DataFrame(all_preds_arr, columns=PRIMARY_LABELS)
    submission.insert(0, 'row_id', all_rows)
    
    assert submission.shape[1] == NUM_CLASSES + 1
    assert submission['row_id'].is_unique
    assert not submission.iloc[:, 1:].isna().any().any()
    submission.iloc[:, 1:] = submission.iloc[:, 1:].clip(0.0, 1.0)
    
    submission.to_csv(_file_name_submission, index=False)
    print(f'Wrote submission.csv: {len(submission)} rows x {submission.shape[1]} cols')
    print(submission.head(3).iloc[:, :6])
