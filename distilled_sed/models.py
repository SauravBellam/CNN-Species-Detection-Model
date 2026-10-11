"""Model definitions for the distilled SED (Model_1).

Adapted from the "Model_1" section of nina2025's Kaggle notebook
"BirdCLEF 2026 EoS.9" (https://www.kaggle.com/code/nina2025/birdclef-2026-eos-9#Model_1),
which in turn follows the BirdCLEF 2025 1st-place SED design.

Change vs the notebook: the EfficientNet-B0 backbone is replaced by this
repo's ResNet-18. The ResNet's global average pool and fc layer are removed so
the attention head sees the full (channel, freq, time) feature map -- pooling
before attention threw away the time axis the attention is supposed to weight.
"""
import torch
import torch.nn as nn
import torchaudio
import torchvision.models as tv_models


class MelSpecTransform(nn.Module):
    """GPU log-mel spectrogram."""

    def __init__(self, sr=32000, n_fft=2048, hop_length=512, n_mels=256, fmin=20, fmax=16000):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=fmin, f_max=fmax, power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def forward(self, waveform):
        return self.db_transform(self.mel_spec(waveform))


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


class PerchTeacher:
    """Frozen Perch v2 via ONNX. Takes 5s waveforms, returns 1536-d embeddings.
    The teacher is never updated -- it provides a stable distillation target."""

    def __init__(self, onnx_path, device_str="cuda", embed_dim=1536):
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if device_str == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self._embed_idx = 1
        for i, o in enumerate(self.session.get_outputs()):
            if o.shape and o.shape[-1] == embed_dim:
                self._embed_idx = i
                break
        print(f"Perch ONNX loaded: embed_idx={self._embed_idx}")

    @torch.no_grad()
    def embed(self, waveforms_5s):
        """waveforms_5s: (B, 160000) float32, returns (B, 1536) embeddings."""
        results = self.session.run(None, {self.input_name: waveforms_5s.cpu().numpy()})
        return torch.from_numpy(results[self._embed_idx]).float()


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
    """ResNet-18 up to layer4 with 1-channel input and NO avgpool / fc,
    so the output keeps its (freq, time) layout: (B, 512, F/32, T/32)."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        r = tv_models.resnet18(weights=weights)
        conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:  # keep ImageNet filters by summing over RGB
            conv1.weight.data = r.conv1.weight.data.sum(dim=1, keepdim=True)
        self.features = nn.Sequential(conv1, r.bn1, r.relu, r.maxpool,
                                      r.layer1, r.layer2, r.layer3, r.layer4)
        self.out_dim = r.fc.in_features

    def forward(self, x):
        return self.features(x)


class BirdSEDModel(nn.Module):
    """ResNet-18 trunk + Model_1's SED head:
    - GeMFreq pooling (learnable, sharper than mean)
    - 512-dim bottleneck with dropout
    - attention-weighted clip logits from frame logits
    - distillation: GAP+Linear branch for MSE to Perch (training only)
    - stop gradient: backbone trains from distillation only

    Ablation switches: freq_pool ("gem"|"mean"), time_pool ("att"|"mean"),
    stop_grad_cls (None -> same as use_distill).
    """

    def __init__(self, num_classes=234, hidden_dim=512, pretrained=True,
                 use_distill=True, embed_dim=1536, stop_grad_cls=None,
                 freq_pool="gem", time_pool="att"):
        super().__init__()
        self.backbone = ResNetTrunk(pretrained=pretrained)
        self.backbone_dim = self.backbone.out_dim
        self.stop_grad_cls = use_distill if stop_grad_cls is None else stop_grad_cls
        self.time_pool = time_pool

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
        if use_distill:
            self.distill_head = DistillHead(self.backbone_dim, embed_dim)

    def forward(self, x, return_framewise=False, return_distill=False):
        h = self.backbone(x)  # (B, C, F, T) -- no pooling before the head
        distill_emb = None
        if return_distill and hasattr(self, "distill_head"):
            distill_emb = self.distill_head(h)

        h_cls = h.detach() if self.stop_grad_cls else h
        h_cls = self.gem_freq(h_cls)                                   # (B, C, T)
        h_cls = self.dense(h_cls.permute(0, 2, 1)).permute(0, 2, 1)    # (B, 512, T)

        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        if self.time_pool == "mean":
            norm_att = torch.full_like(norm_att, 1.0 / norm_att.shape[-1])
        framewise_logits = self.cla(h_cls)
        clip_logits = torch.sum(norm_att * framewise_logits, dim=2)

        fw = framewise_logits.permute(0, 2, 1) if return_framewise else None
        if return_framewise and return_distill:
            return clip_logits, fw, distill_emb
        if return_framewise:
            return clip_logits, fw
        if return_distill:
            return clip_logits, distill_emb
        return clip_logits


class SEDExportWrapper(nn.Module):
    """Inference graph for ONNX: mel -> (clip_logits, framewise_logits).
    The distillation head is not used, so distillation costs nothing at inference."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mel):
        return self.model(mel, return_framewise=True)
