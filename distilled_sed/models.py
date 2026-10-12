"""Model definitions for the distilled SED (Model_1).

Adapted from the "Model_1" section of nina2025's Kaggle notebook
"BirdCLEF 2026 EoS.9" (https://www.kaggle.com/code/nina2025/birdclef-2026-eos-9#Model_1),
which in turn follows the BirdCLEF 2025 1st-place SED design.

Changes vs the notebook:
  * the EfficientNet-B0 backbone is replaced by this repo's ResNet-18. Its
    global average pool and fc layer are removed so the attention head sees the
    full (channel, freq, time) feature map -- pooling before attention threw
    away the time axis the attention is supposed to weight;
  * input is a 3-channel stack of log-mel spectrograms (log-mel, delta,
    delta-delta; see MelStack), so the ImageNet conv1 weights are used as-is;
  * head="pooled" gives the plain ResNet-18 baseline (avgpool + fc);
  * attention_maps() exposes the attention / class-activation maps for the
    heatmap tracking in attention_viz.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision.models as tv_models


class LogMel(nn.Module):
    """Waveform (B, 1, N) or (B, N) -> log-mel in dB (B, 1, n_mels, T), 80 dB floor per window.
    This is what precompute_mels.py stores on disk (float16)."""

    def __init__(self, sr=32000, n_fft=2048, hop_length=512, n_mels=256, fmin=20, fmax=16000, top_db=80):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=fmin, f_max=fmax, power=2.0)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=top_db)

    def forward(self, wav):
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        return self.db(self.mel(wav))


class MelStack(nn.Module):
    """Log-mel dB (B, 1, F, T) -> stacked, standardised input (B, C, F, T).

    stack_mode:
      "delta"  -> [log-mel, delta, delta-delta]   (3 ch; default)
      "single" -> log-mel only                     (1 ch, the notebook's input)
    Each channel is standardised per sample, as the notebook did for its 1 channel.
    """

    def __init__(self, stack_mode="delta"):
        super().__init__()
        assert stack_mode in ("delta", "single"), stack_mode
        self.stack_mode = stack_mode

    @property
    def n_channels(self):
        return 1 if self.stack_mode == "single" else 3

    def forward(self, x):
        x = x.float()
        if self.stack_mode == "delta":
            d1 = torchaudio.functional.compute_deltas(x)
            x = torch.cat([x, d1, torchaudio.functional.compute_deltas(d1)], dim=1)
        mu = x.mean(dim=(2, 3), keepdim=True)
        sd = x.std(dim=(2, 3), keepdim=True)
        return (x - mu) / (sd + 1e-6)


class MelFeatures(nn.Module):
    """Waveform -> model input in one step (LogMel then MelStack). Used at inference,
    where audio arrives raw; training reads the precomputed LogMel output instead."""

    def __init__(self, sr=32000, n_fft=2048, hop_length=512, n_mels=256, fmin=20, fmax=16000,
                 stack_mode="delta", top_db=80):
        super().__init__()
        self.logmel = LogMel(sr, n_fft, hop_length, n_mels, fmin, fmax, top_db)
        self.stack = MelStack(stack_mode)

    @property
    def n_channels(self):
        return self.stack.n_channels

    def forward(self, wav):
        return self.stack(self.logmel(wav))


class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=10, time_mask_param=10, num_freq_masks=1, num_time_masks=2):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, mel):
        for _ in range(self.num_freq_masks):
            mel = self.freq_mask(mel)
        for _ in range(self.num_time_masks):
            mel = self.time_mask(mel)
        return mel


class DistillHead(nn.Module):
    """Projects backbone features to Perch's 1536-d space via GAP + Linear."""

    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)

    def forward(self, feature_map):
        return self.proj(feature_map.mean(dim=[2, 3]))  # (B, C, F, T) -> (B, embed_dim)


class GeMFreqPool(nn.Module):
    """Generalized Mean pooling over frequency. Learnable p starts at 3.0
    (sharper than mean, softer than max). With p fixed at 1 it is a plain mean."""

    def __init__(self, p_init=3.0, eps=1e-6, learnable=True):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)), requires_grad=learnable)
        self.eps = eps

    def forward(self, x):
        p = self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)  # (B, C, F, T) -> (B, C, T)


class ResNetTrunk(nn.Module):
    """ResNet-18 up to layer4 with NO avgpool / fc, so the output keeps its
    (freq, time) layout: (B, 512, F/32, T/32)."""

    def __init__(self, in_chans=3, pretrained=True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        r = tv_models.resnet18(weights=weights)
        conv1 = r.conv1
        if in_chans != 3:
            conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:  # keep ImageNet filters by averaging over RGB
                conv1.weight.data = r.conv1.weight.data.mean(dim=1, keepdim=True).repeat(1, in_chans, 1, 1)
        self.features = nn.Sequential(conv1, r.bn1, r.relu, r.maxpool,
                                      r.layer1, r.layer2, r.layer3, r.layer4)
        self.out_dim = r.fc.in_features

    def forward(self, x):
        return self.features(x)


class BirdSEDModel(nn.Module):
    """ResNet-18 trunk + Model_1's SED head (head="sed"):
    - GeMFreq pooling (learnable, sharper than mean)
    - 512-dim bottleneck with dropout
    - attention-weighted clip logits from frame logits
    - distillation: GAP+Linear branch for MSE to Perch (training only)
    - stop gradient: backbone trains from distillation only

    head="pooled" is the plain ResNet-18 baseline: global avgpool -> dropout -> fc.
    Its "framewise" logits are the fc applied to each time step of the
    freq-averaged map, so clip = mean over time of framewise.

    Ablation switches: freq_pool ("gem"|"mean"), time_pool ("att"|"mean"),
    stop_grad_cls (None -> same as use_distill).
    """

    def __init__(self, num_classes=234, hidden_dim=512, pretrained=True, in_chans=3,
                 head="sed", use_distill=True, embed_dim=1536, stop_grad_cls=None,
                 freq_pool="gem", time_pool="att", dropout=0.2):
        super().__init__()
        self.backbone = ResNetTrunk(in_chans=in_chans, pretrained=pretrained)
        self.backbone_dim = self.backbone.out_dim
        self.head = head
        self.stop_grad_cls = use_distill if stop_grad_cls is None else stop_grad_cls
        self.time_pool = time_pool
        if use_distill:
            self.distill_head = DistillHead(self.backbone_dim, embed_dim)

        if head == "pooled":
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(self.backbone_dim, num_classes)
            return

        self.gem_freq = GeMFreqPool(p_init=3.0 if freq_pool == "gem" else 1.0,
                                    learnable=freq_pool == "gem")
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.att.weight)
        nn.init.xavier_uniform_(self.cla.weight)
        nn.init.zeros_(self.att.bias)
        nn.init.zeros_(self.cla.bias)

    def _heads(self, h):
        """Feature map (B, C, F, T) -> clip logits (B, K), framewise (B, K, T), attention (B, K, T)."""
        h_cls = h.detach() if self.stop_grad_cls else h
        if self.head == "pooled":
            h_t = self.drop(h_cls.mean(dim=2))                          # (B, C, T)
            framewise = self.fc(h_t.permute(0, 2, 1)).permute(0, 2, 1)  # (B, K, T)
            att = torch.full_like(framewise, 1.0 / framewise.shape[-1])
            return framewise.mean(dim=2), framewise, att
        h_cls = self.gem_freq(h_cls)                                    # (B, C, T)
        h_cls = self.dense(h_cls.permute(0, 2, 1)).permute(0, 2, 1)     # (B, 512, T)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        if self.time_pool == "mean":
            norm_att = torch.full_like(norm_att, 1.0 / norm_att.shape[-1])
        framewise = self.cla(h_cls)
        return torch.sum(norm_att * framewise, dim=2), framewise, norm_att

    def forward(self, x, return_framewise=False, return_distill=False):
        h = self.backbone(x)  # (B, C, F, T) -- no pooling before the head
        distill_emb = None
        if return_distill and hasattr(self, "distill_head"):
            distill_emb = self.distill_head(h)
        clip_logits, framewise, _ = self._heads(h)

        fw = framewise.permute(0, 2, 1) if return_framewise else None
        if return_framewise and return_distill:
            return clip_logits, fw, distill_emb
        if return_framewise:
            return clip_logits, fw
        if return_distill:
            return clip_logits, distill_emb
        return clip_logits

    @torch.no_grad()
    def attention_maps(self, x):
        """For heatmaps. Returns
        time_att (B, K, T')   -- attention weights over time per class (uniform for pooled),
        cam      (B, K, F', T') -- class-activation logits per (freq, time) cell: the
                                   classifier applied to every cell of the un-pooled map,
        clip_prob (B, K).
        """
        h = self.backbone(x)
        clip, _, att = self._heads(h)
        cells = h.permute(0, 2, 3, 1)                                   # (B, F, T, C)
        if self.head == "pooled":
            cam = self.fc(cells)
        else:
            cam = F.linear(self.dense(cells), self.cla.weight.squeeze(-1), self.cla.bias)
        return {"time_att": att, "cam": cam.permute(0, 3, 1, 2), "clip_prob": torch.sigmoid(clip)}


class SEDExportWrapper(nn.Module):
    """Inference graph for ONNX: features -> (clip_logits, framewise_logits).
    The distillation head is not used, so distillation costs nothing at inference."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mel):
        return self.model(mel, return_framewise=True)
