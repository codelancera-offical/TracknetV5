import torch
import torch.nn as nn
from ..builder import BACKBONES
from ..basic import BasicConvBlock as ConvBlock

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

class MultiScaleResidualBlock(nn.Module):
    """
    多尺度残差快
    并行使用1x1 3x3 5x5卷积核，然后特征融合
    带有一个残差连接
    """
    def __init__(self, in_dim, out_dim):
        super(MultiScaleResidualBlock, self).__init__()

        # 路径1： 1x1 -> 3x3
        self.conv_1x1_path = nn.Sequential(
            ConvBlock(in_dim, out_dim, k=1),
            ConvBlock(out_dim, out_dim, k=3)
        )

        # 路径2： 3x3 -> 3x3 (标准路径，也用于残差连接)
        self.conv_3x3_path = nn.Sequential(
            ConvBlock(in_dim, out_dim, k=3),
            ConvBlock(out_dim, out_dim, k=3)
        )

        # 路径3： 5x5 -> 3x3 (更大感受野)
        self.conv_5x5_path = nn.Sequential(
            ConvBlock(in_dim, out_dim, k=5),
            ConvBlock(out_dim, out_dim, k=3)
        )

        # 融合卷积：将3*out_dim 融合回 out_dim
        self.fusion_conv = ConvBlock(out_dim * 3, out_dim, 3)

    def forward(self, x):
        # 1. 兵分三路
        path1_out = self.conv_1x1_path(x)
        path2_out = self.conv_3x3_path(x)
        path3_out = self.conv_5x5_path(x)

        # 2. 特征融合
        x_fused = torch.cat([path1_out, path2_out, path3_out], dim=1) # C维度上拼接
        x_out = self.fusion_conv(x_fused)

        # 3. 残差连接
        x_out = x_out + path2_out

        return x_out

# ==================== Backbone 主体 ====================

@BACKBONES.register_module
class TrackNetV3Backbone(nn.Module):
    """
    TrackNetV3 的 Backbone 部分 (编码器)
    它是一个标准的 U-Net 下采样路径，负责提取多尺度特征。
    """
    def __init__(self, in_channels=9):
        super().__init__()
        # 下采样层 这里计算量是不是少了很多？ 速度有没有提升？（层数貌似变少了）
        self.down_block_1 = MultiScaleResidualBlock(in_channels, 64)
        self.down_block_2 = MultiScaleResidualBlock(64, 128)
        self.down_block_3 = MultiScaleResidualBlock(128, 256)

        # 池化层
        self.pool = nn.MaxPool2d((2, 2), stride=(2, 2))

        # 瓶颈层
        self.bottleneck = Triple2DConv(256, 512)

    def forward(self, x):
        """
        输入: 图像张量, 形状为 [B, C_in, H, W]
        输出: 一个包含各层特征图的字典，用于跳跃连接。
        """
        features = {}
        
        x1 = self.down_block_1(x)
        x = self.pool(x1)
        
        x2 = self.down_block_2(x)
        x = self.pool(x2)

        x3 = self.down_block_3(x)
        x = self.pool(x3)

        bottleneck_out = self.bottleneck(x)
        
        return {
            'skip1': x1,
            'skip2': x2,
            'skip3': x3,
            'bottleneck': bottleneck_out
        }

# ==================== 测试代码 ====================
if __name__ == "__main__":
    batch_size = 2
    input_height = 288
    input_width = 512
    in_channels = 9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用的设备: {device}")
    
    model = TrackNetV3Backbone(in_channels=in_channels).to(device)
    model.eval()
    
    test_tensor = torch.randn(batch_size, in_channels, input_height, input_width).to(device)
    print(f"输入张量形状: {test_tensor.shape}")

    expected_shapes = {
        'skip1': (batch_size, 64, input_height, input_width),
        'skip2': (batch_size, 128, input_height // 2, input_width // 2),
        'skip3': (batch_size, 256, input_height // 4, input_width // 4),
        'bottleneck': (batch_size, 512, input_height // 8, input_width // 8)
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