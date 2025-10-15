import torch
import torch.nn as nn
from ..builder import BACKBONES

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

# ==================== Backbone 主体 ====================

@BACKBONES.register_module
class TrackNetV3Backbone(nn.Module):
    """
    TrackNetV3 的 Backbone 部分 (编码器)
    它是一个标准的 U-Net 下采样路径，负责提取多尺度特征。
    """
    def __init__(self, in_channels=9):
        super().__init__()
        self.down_block_1 = Double2DConv(in_channels, 64)
        self.down_block_2 = Double2DConv(64, 128)
        self.down_block_3 = Triple2DConv(128, 256)
        self.pool = nn.MaxPool2d((2, 2), stride=(2, 2))
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