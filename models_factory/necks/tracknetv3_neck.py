import torch
import torch.nn as nn
from ..builder import NECKS

# ==================== 建筑模块 ====================
# 为了让此脚本能独立运行，我们在此处包含 TrackNetV3 所需的基础模块定义

class Conv2DBlock(nn.Module):
    """ Conv2D + BN + ReLU """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding='same', bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Double2DConv(nn.Module):
    """ Conv2DBlock x 2 """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x
    
class Triple2DConv(nn.Module):
    """ Conv2DBlock x 3 """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)
        self.conv_3 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x
        
# ==================== Neck 主体 ====================

@NECKS.register_module
class TrackNetV3Neck(nn.Module):
    """
    TrackNetV3 的 Neck 部分 (解码器)
    它接收来自Backbone的多尺度特征，通过上采样和跳跃连接逐步恢复特征图分辨率。
    """
    def __init__(self):
        super().__init__()
        self.up_block_1 = Triple2DConv(512 + 256, 256) # 输入: bottleneck(512) + skip3(256)
        self.up_block_2 = Double2DConv(256 + 128, 128) # 输入: 上一层(256) + skip2(128)
        self.up_block_3 = Double2DConv(128 + 64, 64)   # 输入: 上一层(128) + skip1(64)

    def forward(self, features):
        """
        输入: 来自Backbone的特征字典
        输出: 精炼后的高分辨率特征图
        """
        x3 = features['skip3']
        x2 = features['skip2']
        x1 = features['skip1']
        x = features['bottleneck']

        x = nn.Upsample(scale_factor=2)(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up_block_1(x)
        
        x = nn.Upsample(scale_factor=2)(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up_block_2(x)
        
        x = nn.Upsample(scale_factor=2)(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_block_3(x)
        
        return x

# ==================== 测试代码 ====================
if __name__ == "__main__":
    batch_size = 2
    height = 288
    width = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"使用的设备: {device}")

    model = TrackNetV3Neck().to(device)
    model.eval()

    mock_features = {
        'skip1': torch.randn(batch_size, 64, height, width).to(device),
        'skip2': torch.randn(batch_size, 128, height // 2, width // 2).to(device),
        'skip3': torch.randn(batch_size, 256, height // 4, width // 4).to(device),
        'bottleneck': torch.randn(batch_size, 512, height // 8, width // 8).to(device)
    }
    
    print("\n--- 模拟输入特征形状 ---")
    for name, tensor in mock_features.items():
        print(f"  - {name}: {tensor.shape}")

    expected_output_shape = (batch_size, 64, height, width)
    
    print(f"\n预期的最终输出形状: {expected_output_shape}")
    print("\n--- 开始测试 ---")

    with torch.no_grad():
        output_tensor = model(mock_features)

    actual_output_shape = output_tensor.shape
    print(f"实际的最终输出形状: {actual_output_shape}")
    
    print("\n--- 测试结论 ---")
    if actual_output_shape == expected_output_shape:
        print("✅ 测试通过")
    else:
        print("❌ 测试不通过")