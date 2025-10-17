import torch
import torch.nn as nn
from ..builder import MODELS, build_backbone, build_neck, build_head

@MODELS.register_module
class InpaintNet(nn.Module):
    """
    InpaintNet 模型总装
    它接收轨迹点(x)和掩码(m)作为输入，处理后送入骨干网络。
    """
    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)

    def forward(self, x, m):
        # 1. 预处理：合并输入并调整维度
        # x: [B, L, 2], m: [B, L, 1]
        input_tensor = torch.cat([x, m], dim=2) # -> [B, L, 3]
        input_tensor = input_tensor.permute(0, 2, 1) # -> [B, 3, L]

        # 2. 核心流程
        features = self.backbone(input_tensor)
        refined_features = self.neck(features)
        output = self.head(refined_features) # -> [B, L, 2]
        return output
