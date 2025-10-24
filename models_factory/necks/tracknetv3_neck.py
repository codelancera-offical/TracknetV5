import torch
import torch.nn as nn
from ..builder import NECKS
from ..basic import BasicConvBlock as ConvBlock
from ..basic import ChannelAttention as CAM
# ==================== 建筑模块 ====================
# 为了让此脚本能独立运行，我们在此处包含 TrackNetV3 所需的基础模块定义

class Double2DConv(nn.Module):
    """ ConvBlock x 2 """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_1 = ConvBlock(in_dim, out_dim)
        self.conv_2 = ConvBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x


class Triple2DConv(nn.Module):
    """ ConvBlock x 3 """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_1 = ConvBlock(in_dim, out_dim)
        self.conv_2 = ConvBlock(out_dim, out_dim)
        self.conv_3 = ConvBlock(out_dim, out_dim)

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
        self.up_block_1 = Double2DConv(512 + 256, 256) # 输入: bottleneck(512) + skip3(256)
        self.up_block_2 = Double2DConv(256 + 128, 128) # 输入: 上一层(256) + skip2(128)
        self.up_block_3 = Double2DConv(128 + 64, 64)   # 输入: 上一层(128) + skip1(64)


        # 注意力模块（跳跃连接路径）
        self.cam0_1 = CAM(in_planes=256)
        self.cam0_2 = CAM(in_planes=128)
        self.cam0_3 = CAM(in_planes=64)

        # 注意力模块（上采样路径）
        self.cam1 = CAM(in_planes=256)
        self.cam2 = CAM(in_planes=128)
        self.cam3 = CAM(in_planes=64)

        # 上采样层
        self.upsample = nn.Upsample(scale_factor=2)



    def forward(self, features):
        """
        输入: 来自Backbone的特征字典
        输出: 精炼后的高分辨率特征图
        """
        x3 = features['skip3'] # 64
        x2 = features['skip2'] # 128
        x1 = features['skip1'] # 256
        x = features['bottleneck']

        # L1
        x = nn.Upsample(scale_factor=2)(x)
        x3_att = x3 * self.cam0_1(x3)
        x = torch.cat([x, x3_att], dim=1)
        x = self.up_block_1(x)
        x = x * self.cam1(x)
        
        x = nn.Upsample(scale_factor=2)(x)
        x2_att = x2 * self.cam0_2(x2)
        x = torch.cat([x, x2_att], dim=1)
        x = self.up_block_2(x)
        x = x * self.cam2(x)
        
        x = nn.Upsample(scale_factor=2)(x)
        x1_att = x1 * self.cam0_3(x1)
        x = torch.cat([x, x1_att], dim=1)
        x = self.up_block_3(x)
        x = x * self.cam3(x)
        
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