"""Ablation variants and the hypothesis each one tests.

RESEARCH GOAL: find the cheapest model we can train and evaluate that still
performs well -- cheap and fast enough to run 24/7 on site. Every variant is
judged on quality (many metrics, not just ROC-AUC) AND on cost
(benchmark_cost.py: params, size, CPU ms per 5 s window). A component earns
its place only if what it buys is worth what it costs at inference.

Pipeline under test: Perch v2 pseudo-labels every 5 s window offline
(pseudo_label.py), then a ResNet-18 with Model_1's attention SED head is
trained on 3-channel stacked log-mel spectrograms, distilled from cached Perch
embeddings. Each training variant changes ONE thing relative to `full`.
"""

# name -> SED_OVERRIDES for model1_sed.py ("script" variants run their own script)
TRAIN_VARIANTS = {
    # baselines
    "resnet_baseline": {"HEAD": "pooled", "USE_PERCH_DISTILL": False},
    "perch_attn_probe": {"script": "perch_probe"},
    # full model
    "full": {},
    # one-change ablations
    "no_distill": {"USE_PERCH_DISTILL": False},
    "distill_no_stopgrad": {"STOP_GRAD_CLS": False},
    "no_pseudo_focal": {"USE_PSEUDO_FOCAL": False},
    "no_pseudo_sc": {"USE_PSEUDO_SC": False},
    "single_channel_mel": {"STACK_MODE": "single"},
    "multires_mel": {"STACK_MODE": "multires"},
    "mean_freq_pool": {"FREQ_POOL": "mean"},
    "mean_time_pool": {"TIME_POOL": "mean"},
    "no_mixup": {"USE_FOCAL_MIXUP": False, "USE_FOCAL_SC_MIXUP": False},
    "no_soundscape_data": {"USE_LABELED_SC": False, "USE_FOCAL_SC_MIXUP": False,
                           "USE_PSEUDO_SC": False},
    "no_specaug": {"NUM_FREQ_MASKS": 0, "NUM_TIME_MASKS": 0},
    "no_rare_upsample": {"MIN_SAMPLE": 0},
}

# Post-hoc variants computed from the `full` run's OOF (no retraining, ~zero inference cost)
POSTHOC_VARIANTS = {
    "head_clip_only": "clip",           # attention-pooled clip logits only
    "head_framemax_only": "fmax",       # max over frame logits only
    "full+gauss_smooth": "smooth",      # 5-tap smoothing across neighbouring windows
    "full+tax_smooth": "tax",           # pull toward genus / class means (notebook TAX_SMOOTHING)
    "full+site_hour_prior": "prior",    # fold-safe site/hour species prior in logit space
    "full+all_posthoc": "all",          # prior -> gauss smooth -> tax smooth
}

# (variant, metric the change should move most, direction, rationale)
HYPOTHESES = [
    ("resnet_baseline", "lrap", "drop",
     "A plain pooled ResNet loses the time axis, so it ranks species within a window worse "
     "(LRAP) and is less calibrated than the attention SED head at the same cost."),
    ("resnet_baseline", "ece", "rise",
     "Without distillation or frame-level heads the baseline overfits the small label set."),
    ("perch_attn_probe", "auc_rare", "rise",
     "Perch has seen far more data; the frozen teacher should beat the student on rare "
     "species, at many times the inference cost (see benchmark_cost.py)."),
    ("no_pseudo_focal", "pr_auc_macro", "drop",
     "Clip-level labels mark every 5 s window as positive even when the bird is silent; "
     "Perch-localised targets remove those false positives, mainly helping precision (PR-AUC)."),
    ("no_pseudo_sc", "auc_soundscape_macro", "drop",
     "Unlabelled soundscapes are in-domain; their pseudo-labels should help soundscape AUC "
     "and calibration most."),
    ("single_channel_mel", "auc_Insecta", "drop",
     "Delta channels encode modulation / onsets; insects and frogs are defined by "
     "repetition rate, so dropping them should hurt non-bird taxa more."),
    ("multires_mel", "auc_soundscape_macro", "~same",
     "Three STFT resolutions cost 3x the feature time; expect little gain over deltas."),
    ("no_distill", "auc_rare", "drop",
     "Perch embeddings carry knowledge about species with few focal clips."),
    ("no_distill", "ece", "rise",
     "Without the teacher regulariser the student overfits, giving worse calibration."),
    ("distill_no_stopgrad", "auc_soundscape_macro", "~same",
     "Letting the classification loss update the backbone should matter little."),
    ("mean_freq_pool", "auc_Insecta", "drop",
     "GeM sharpens toward narrow bands; insects/frogs are narrow-band."),
    ("mean_time_pool", "lrap", "drop",
     "Attention pooling picks the frames with the call; uniform pooling dilutes short calls."),
    ("no_soundscape_data", "auc_soundscape_macro", "drop",
     "Soundscape windows match the deployment domain."),
    ("no_mixup", "f1_macro", "drop",
     "Mixup teaches multi-species windows, so multi-label F1 should drop more than AUC."),
    ("no_specaug", "auc_soundscape_macro", "small drop",
     "SpecAugment is a generic regulariser; expect a small uniform drop."),
    ("no_rare_upsample", "auc_rare", "drop",
     "Upsampling only touches species with <20 clips."),
    ("head_clip_only", "lrap", "drop",
     "Clip and frame-max heads make different errors; the 50/50 blend should beat either."),
    ("head_framemax_only", "ece", "rise",
     "Max over frames is biased upward, so frame-max-only should be over-confident."),
    ("full+gauss_smooth", "temporal_jitter", "drop",
     "Smoothing across neighbouring windows mainly reduces prediction jitter."),
    ("full+tax_smooth", "auc_rare", "rise",
     "Rare species borrow evidence from relatives in the same genus."),
    ("full+site_hour_prior", "auc_soundscape_macro", "rise",
     "Where and when a recording was made is a strong, free signal for which species "
     "can be present -- expected to be the best gain per unit of inference cost."),
    ("full+all_posthoc", "auc_soundscape_macro", "rise",
     "The free post-processing steps stack."),
]
