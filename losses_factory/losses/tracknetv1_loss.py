import torch
import torch.nn as nn
from ..builder import LOSSES

@LOSSES.register_module
class TrackNetV1Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        loss = self.criterion(logits, targets)
        return loss


if __name__ == "__main__":
    print("--- 测试 TrackNetV1Loss (基于256灰度等级分类原理) ---")

    # 定义模拟参数
    B, C, H, W = 4, 256, 640, 360  # 批大小=4, 类别/灰度等级=256, 尺寸=64x64
    
    # 1. 初始化损失函数
    loss_fn = TrackNetV1Loss()
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