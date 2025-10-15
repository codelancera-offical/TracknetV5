import torch
import torch.nn as nn
from ..builder import HEADS

@HEADS.register_module
class TrackNetV3Head(nn.Module):
    """
    TrackNetV3 的 Head 部分 (预测头)
    它通过一个1x1卷积将Neck输出的特征图通道数映射到最终的输出维度，并用Sigmoid激活。
    """
    def __init__(self, in_channels=64, out_channels=3):
        super().__init__()
        self.predictor = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        输入: 来自Neck的精炼特征图, 形状为 [B, in_channels, H, W]
        输出: 最终预测结果 (如热力图), 形状为 [B, out_channels, H, W]
        """
        x = self.predictor(x)
        x = self.sigmoid(x)
        return x

# ==================== 测试代码 ====================
if __name__ == "__main__":
    batch_size = 4
    in_channels = 64  # Neck 输出的通道数
    out_channels = 3  # TrackNetV3 默认输出3通道
    height = 288
    width = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用的设备: {device}")

    model = TrackNetV3Head(in_channels=in_channels, out_channels=out_channels).to(device)
    model.eval()

    mock_input_tensor = torch.randn(batch_size, in_channels, height, width).to(device)
    print(f"\n模拟输入张量形状: {mock_input_tensor.shape}")

    expected_output_shape = (batch_size, out_channels, height, width)
    print(f"预期输出张量形状: {expected_output_shape}")

    print("\n--- 开始测试 ---")

    with torch.no_grad():
        output_tensor = model(mock_input_tensor)

    actual_output_shape = output_tensor.shape
    print(f"实际输出张量形状: {actual_output_shape}")

    print("\n--- 测试结论 ---")
    if actual_output_shape == expected_output_shape:
        print("✅ 测试通过")
    else:
        print("❌ 测试不通过")