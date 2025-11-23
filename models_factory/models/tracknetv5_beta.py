import torch.nn as nn
import torch
from ..builder import MODELS, build_backbone, build_neck, build_head
from ..basic import MDD

@MODELS.register_module
class TrackNetV5Beta(nn.Module):
    def __init__(self, backbone, neck, head):
        super().__init__()
        self.mdd = MDD()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)

    def forward(self, x):
        mvdr_attention = self.mdd(x)
        features = self.backbone(x)
        refined_features = self.neck(features)
        logits = self.head(refined_features, mvdr_attention)
        return logits

