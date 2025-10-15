import torch
import torch.nn as nn
from ..builder import BACKBONES

# ==================== 建筑模块 ====================
class Conv1DBlock(nn.Module):
    """ Conv1D + LeakyReLU"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding='same', bias=True)
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class Double1DConv(nn.Module):
    """ Conv1DBlock x 2"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_1 = Conv1DBlock(in_dim, out_dim)
        self.conv_2 = Conv1DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

# ==================== Backbone 主体 ====================
@BACKBONES.register_module
class InpaintNetBackbone(nn.Module):
    """
    InpaintNet 的 Backbone 部分 (1D 编码器)
    它接收 1D 序列数据，通过 1D 卷积提取特征。
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.down_1 = Conv1DBlock(in_channels, 32)
        self.down_2 = Conv1DBlock(32, 64)
        self.down_3 = Conv1DBlock(64, 128)
        self.bottleneck = Double1DConv(128, 256)

    def forward(self, x):
        """
        输入: 1D 序列张量, 形状为 [B, C_in, L]
        输出: 包含各层特征的字典，用于跳跃连接。
        """
        x1 = self.down_1(x)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        bottleneck_out = self.bottleneck(x3)

        return {
            'skip1': x1,
            'skip2': x2,
            'skip3': x3,
            'bottleneck': bottleneck_out
        }

# ==================== 测试代码 ====================
if __name__ == "__main__":
    batch_size = 4
    sequence_length = 100
    in_channels = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"使用的设备: {device}")

    model = InpaintNetBackbone(in_channels=in_channels).to(device)
    model.eval()

    test_tensor = torch.randn(batch_size, in_channels, sequence_length).to(device)
    print(f"输入张量形状: {test_tensor.shape}")

    expected_shapes = {
        'skip1': (batch_size, 32, sequence_length),
        'skip2': (batch_size, 64, sequence_length),
        'skip3': (batch_size, 128, sequence_length),
        'bottleneck': (batch_size, 256, sequence_length)
    }

    print("\n--- 开始测试 ---")
    with torch.no_grad():
        output_features = model(test_tensor)

    test_passed = True
    print("检查输出特征图的形状:")
    for name, feature_map in output_features.items():
        actual_shape = feature_map.shape
        expected_shape = expected_shapes[name]
        if actual_shape == expected_shape:
            print(f"  - 特征 '{name}': 形状 {actual_shape} 符合预期。")
        else:
            print(f"  - [失败] 特征 '{name}': 形状 {actual_shape}，但预期为 {expected_shape}。")
            test_passed = False

    print("\n--- 测试结论 ---")
    if test_passed:
        print("✅ 测试通过")
    else:
        print("❌ 测试不通过")