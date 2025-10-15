import torch
import torch.nn as nn
from ..builder import HEADS

@HEADS.register_module
class InpaintNetHead(nn.Module):
    """
    InpaintNet 的 Head 部分 (1D 预测头)
    将精炼后的 1D 特征序列转换为最终的坐标预测。
    """
    def __init__(self, in_channels=32, out_channels=2):
        super().__init__()
        self.predictor = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        输入: 来自Neck的精炼特征序列, 形状为 [B, C, L]
        输出: 最终预测坐标, 形状为 [B, L, 2]
        """
        x = self.predictor(x)
        x = self.sigmoid(x)
        x = x.permute(0, 2, 1) # [B, C, L] -> [B, L, C]
        return x

# ==================== 测试代码 ====================
if __name__ == "__main__":
    batch_size = 4
    sequence_length = 100
    in_channels = 32  # Neck 输出的通道数
    out_channels = 2  # 最终输出2个坐标值 (x, y)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用的设备: {device}")

    model = InpaintNetHead(in_channels=in_channels, out_channels=out_channels).to(device)
    model.eval()

    mock_input_tensor = torch.randn(batch_size, in_channels, sequence_length).to(device)
    print(f"\n模拟输入张量形状: {mock_input_tensor.shape}")

    expected_output_shape = (batch_size, sequence_length, out_channels)
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