import torch.nn as nn

from .distilled_sed.models import BirdSEDModel


class BirdResNet(nn.Module):
    """ResNet-18 + attention SED head (see distilled_sed/models.py).

    Previously ResNet-18 -> global avgpool -> fc. The avgpool collapsed the
    time axis, so there was nothing left for an attention head to weight; the
    trunk now hands its (freq, time) feature map straight to GeM frequency
    pooling and the attention head. Returns clip logits, same as before.
    """

    def __init__(self, num_classes, dropout=0.2):
        super().__init__()
        self.model = BirdSEDModel(num_classes=num_classes, use_distill=False)

    def forward(self, x, return_framewise=False):
        return self.model(x, return_framewise=return_framewise)
