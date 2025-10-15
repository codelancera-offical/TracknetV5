import torch
import torch.nn as nn
from ..builder import MODELS, build_backbone, build_neck, build_head

@MODELS.register_module
class TrackNetV3(nn.Module):
    """
    TrackNetV3 模型总装
    它通过配置文件，使用 builder 动态构建 backbone, neck, 和 head, 并将它们串联起来。
    """
    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)

    def forward(self, x):
        features = self.backbone(x)
        refined_features = self.neck(features)
        output = self.head(refined_features)
        return output

