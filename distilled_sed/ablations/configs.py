"""Ablation variants for the distilled SED (Model_1) and the hypothesis each one tests.

Each training variant changes ONE thing relative to `full` (the notebook's
original Model_1 config). Post-hoc variants (output head mode, temporal
smoothing) need no retraining: evaluate.py derives them from the saved OOF.

Every hypothesis names the metric it should move most, so the paper can say
"X mainly buys Y" instead of only "X adds +0.00Z AUC".
"""

# name -> SED_OVERRIDES passed to model1_sed.py
TRAIN_VARIANTS = {
    "full": {},
    "no_distill": {"USE_PERCH_DISTILL": False},
    "distill_no_stopgrad": {"STOP_GRAD_CLS": False},
    "mean_freq_pool": {"FREQ_POOL": "mean"},
    "mean_time_pool": {"TIME_POOL": "mean"},
    "no_mixup": {"USE_FOCAL_MIXUP": False, "USE_FOCAL_SC_MIXUP": False},
    "no_soundscape_data": {"USE_LABELED_SC": False, "USE_FOCAL_SC_MIXUP": False},
    "no_specaug": {"NUM_FREQ_MASKS": 0, "NUM_TIME_MASKS": 0},
    "no_rare_upsample": {"MIN_SAMPLE": 0},
}

# Post-hoc variants computed from the `full` run's OOF
POSTHOC_VARIANTS = {
    "head_clip_only": "clip",      # attention-pooled clip logits only
    "head_framemax_only": "fmax",  # max over frame logits only
    "full+gauss_smooth": "smooth", # notebook's inference-time 5-tap smoothing
}

# Hypotheses: (variant, metric the change should hurt most, direction, rationale)
HYPOTHESES = [
    ("no_distill", "auc_rare", "drop",
     "Perch embeddings carry knowledge about species with few focal clips, "
     "so removing distillation should hurt rare-species AUC more than common-species AUC."),
    ("no_distill", "ece", "rise",
     "Without the teacher regulariser the student overfits the small label set, "
     "giving worse calibration (higher ECE) even if ranking barely moves."),
    ("distill_no_stopgrad", "auc_soundscape_macro", "~same",
     "Letting the classification loss update the backbone should matter little; "
     "the stop-gradient mainly keeps the backbone a clean Perch imitator."),
    ("mean_freq_pool", "auc_Insecta", "drop",
     "GeM sharpens toward narrow bands; insects/frogs are narrow-band, so plain "
     "mean pooling should hurt non-bird taxa more than birds."),
    ("mean_time_pool", "lrap", "drop",
     "Attention pooling picks the frames where the call is; uniform pooling dilutes "
     "short calls in 5 s windows, hurting the per-window ranking (LRAP) and recall@F1."),
    ("no_soundscape_data", "auc_soundscape_macro", "drop",
     "Soundscape windows match the test domain; dropping them should cost the most "
     "domain-shift-sensitive metric (soundscape AUC) and calibration."),
    ("no_mixup", "f1_macro", "drop",
     "Mixup teaches multi-species windows, so without it multi-label F1 should drop "
     "more than AUC."),
    ("no_specaug", "auc_soundscape_macro", "small drop",
     "SpecAugment is a generic regulariser; expect a small uniform drop."),
    ("no_rare_upsample", "auc_rare", "drop",
     "Upsampling only touches species with <20 clips, so the effect should be "
     "concentrated in rare-species AUC with ~0 change for common species."),
    ("head_clip_only", "lrap", "drop",
     "Clip and frame-max heads make different errors; the 50/50 blend should beat either."),
    ("head_framemax_only", "ece", "rise",
     "Max over frames is biased upward, so frame-max-only should be over-confident."),
    ("full+gauss_smooth", "temporal_jitter", "drop",
     "Smoothing across neighbouring windows should mainly reduce prediction jitter "
     "and improve AUC slightly, at zero inference cost."),
]
