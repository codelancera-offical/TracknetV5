import torch.nn as nn
import torch
from models_factory.basic import mv_dr_attention
from ..builder import MODELS, build_backbone, build_neck, build_head

from ..basic import MDD

@MODELS.register_module
class TrackNetV2MDD(nn.Module):
    def __init__(self, backbone, neck, head):
        super().__init__()
        self.mdd = MDD()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)
    
    def forward(self, x):
        mdd_attention_maps = self.mdd(x)
        
        F1 = x[:, 0:3]
        F2 = x[:, 3:6]
        F3 = x[:, 6:9]
        att_12 = mdd_attention_maps[:, 0:2]
        att_23 = mdd_attention_maps[:, 2:4]

        x = torch.cat([
            F1,
            att_12,
            F2,
            att_23,
            F3
        ], dim=1)

        features = self.backbone(x)
        refined_features = self.neck(features)
        logits = self.head(refined_features)

        return logits