| variant | metric | predicted | Δ vs full | biggest-Δ metric | verdict |
|---|---|---|---|---|---|
| resnet_baseline | lrap | drop | +0.0563 | auc_Mammalia | not supported |
| | | | | | _A plain pooled ResNet loses the time axis, so it ranks species within a window worse (LRAP) and is less calibrated than the attention SED head at the same cost._ |
| resnet_baseline | ece | rise | +0.0419 | auc_Mammalia | supported |
| | | | | | _Without distillation or frame-level heads the baseline overfits the small label set._ |
| perch_attn_probe | auc_rare | rise | -0.0901 | auc_Mammalia | not supported |
| | | | | | _Perch has seen far more data; the frozen teacher should beat the student on rare species, at many times the inference cost (see benchmark_cost.py)._ |
| no_pseudo_focal | pr_auc_macro | drop | +0.0040 | auc_Mammalia | not supported |
| | | | | | _Clip-level labels mark every 5 s window as positive even when the bird is silent; Perch-localised targets remove those false positives, mainly helping precision (PR-AUC)._ |
| no_pseudo_sc | auc_soundscape_macro | drop | +0.0600 | auc_Mammalia | not supported |
| | | | | | _Unlabelled soundscapes are in-domain; their pseudo-labels should help soundscape AUC and calibration most._ |
| single_channel_mel | auc_Insecta | drop | -0.0019 | auc_Mammalia | supported |
| | | | | | _Delta channels encode modulation / onsets; insects and frogs are defined by repetition rate, so dropping them should hurt non-bird taxa more._ |
| no_distill | auc_rare | drop | +0.0325 | auc_Amphibia | not supported |
| | | | | | _Perch embeddings carry knowledge about species with few focal clips._ |
| no_distill | ece | rise | +0.0413 | auc_Amphibia | supported |
| | | | | | _Without the teacher regulariser the student overfits, giving worse calibration._ |
| distill_no_stopgrad | auc_soundscape_macro | ~same | -0.0172 | auc_common | not supported |
| | | | | | _Letting the classification loss update the backbone should matter little._ |
| mean_freq_pool | auc_Insecta | drop | +0.0147 | auc_Mammalia | not supported |
| | | | | | _GeM sharpens toward narrow bands; insects/frogs are narrow-band._ |
| mean_time_pool | lrap | drop | -0.0108 | auc_Amphibia | supported |
| | | | | | _Attention pooling picks the frames with the call; uniform pooling dilutes short calls._ |
| no_soundscape_data | auc_soundscape_macro | drop | -0.0486 | auc_Amphibia | supported |
| | | | | | _Soundscape windows match the deployment domain._ |
| no_mixup | f1_macro | drop | -0.0471 | auc_Aves | supported |
| | | | | | _Mixup teaches multi-species windows, so multi-label F1 should drop more than AUC._ |
| no_specaug | auc_soundscape_macro | small drop | -0.0007 | auc_Mammalia | supported |
| | | | | | _SpecAugment is a generic regulariser; expect a small uniform drop._ |
| no_rare_upsample | auc_rare | drop | -0.0043 | auc_Mammalia | supported |
| | | | | | _Upsampling only touches species with <20 clips._ |
| head_clip_only | lrap | drop | +0.0058 | f1_macro | not supported |
| | | | | | _Clip and frame-max heads make different errors; the 50/50 blend should beat either._ |
| head_framemax_only | ece | rise | +0.0360 | ece | supported |
| | | | | | _Max over frames is biased upward, so frame-max-only should be over-confident._ |
| full+gauss_smooth | temporal_jitter | drop | -0.0313 | temporal_jitter | supported |
| | | | | | _Smoothing across neighbouring windows mainly reduces prediction jitter._ |
| full+tax_smooth | auc_rare | rise | +0.0629 | auc_Insecta | supported |
| | | | | | _Rare species borrow evidence from relatives in the same genus._ |
| full+site_hour_prior | auc_soundscape_macro | rise | +0.0204 | ece | supported |
| | | | | | _Where and when a recording was made is a strong, free signal for which species can be present -- expected to be the best gain per unit of inference cost._ |
| full+all_posthoc | auc_soundscape_macro | rise | +0.1266 | auc_Insecta | supported |
| | | | | | _The free post-processing steps stack._ |
