# 文件路径: Ball-Tracking/losses/tracknetv1_loss.py

import torch
import torch.nn as nn
from ..builder import LOSSES

@LOSSES.register_module
class UTrackNetV1Loss(nn.Module):
    # ✨ 在 __init__ 中增加一个 class_weights 参数
    def __init__(self, class_weights: list = None, ignore_index: int = -100):
        super().__init__()
        
        weight_tensor = None
        if class_weights is not None:
            # 如果配置中传入了权重列表，就把它转换成Tensor
            weight_tensor = torch.FloatTensor(class_weights)
            
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算损失值。

        参数:
            logits (torch.Tensor):
                来自模型头部的原始输出，代表每个像素在256个灰度等级上的得分。
                期望形状: [B, C, H, W]，其中 C 是灰度等级数，通常为 256。

            targets (torch.Tensor):
                真实的灰度标签图 (Ground Truth Grayscale Map)。
                期望形状: [B, H, W]，其中每个像素的值是其真实的灰度等级 [0, 255]。
                期望数据类型: torch.long (int64)。

        返回:
            torch.Tensor: 计算出的交叉熵损失值，一个标量张量。
        """
        loss = self.criterion(logits, targets)
        return loss

# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("--- 测试 TrackNetV1Loss (基于256灰度等级分类原理) ---")

    # 定义模拟参数
    B, C, H, W = 4, 256, 640, 360  # 批大小=4, 类别/灰度等级=256, 尺寸=64x64
    
    # 1. 初始化损失函数
    loss_fn = UTrackNetV1Loss()
    print(f"损失函数已初始化: {loss_fn.criterion}")
    
    # 2. 创建模拟输入
    # 模型的输出 logits, 形状 [B, 256, H, W]
    mock_logits = torch.randn(B, C, H, W)
    # 真实的灰度标签图, 形状 [B, H, W], 值为 0 到 255
    max_value = C
    mock_targets = torch.randint(0, max_value, (B, H, W), dtype=torch.long)
    
    print(f"\n模拟 Logits 形状: {mock_logits.shape}")
    print(f"模拟 Targets 形状: {mock_targets.shape}")
    print(f"Targets 数据类型: {mock_targets.dtype}")

    # 3. 计算损失
    try:
        loss_value = loss_fn(mock_logits, mock_targets)
        print(f"\n计算得到的损失值: {loss_value.item():.4f}")
        print(f"损失值是一个标量: {loss_value.dim() == 0}")
        print("\n✅ 测试通过：损失函数成功处理了符合原理的输入维度。")
    except Exception as e:
        print(f"\n❌ 测试失败. 错误: {e}")