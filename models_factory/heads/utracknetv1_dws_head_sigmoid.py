import torch
import torch.nn as nn
from ..builder import BACKBONES, HEADS


class DepthwiseSeparableConvBlock(nn.Module):
    """
    深度可分离卷积块。
    
    将一个标准卷积分解为：
    1. 深度卷积 (Depthwise Convolution)
    2. 逐点卷积 (Pointwise Convolution)
    
    这可以显著减少参数量和计算成本。
    """
    def __init__(self, in_channels, out_channels, stride=1):
        """
        初始化深度可分离卷积块。
        
        参数:
        - in_channels (int): 输入特征图的通道数。
        - out_channels (int): 输出特征图的通道数。
        - stride (int): 卷积步长，默认为 1。
        """
        super().__init__()
        
        self.depthwise_conv = nn.Sequential(
            # --- 1. 深度卷积 ---
            # 对每个输入通道应用一个独立的 3x3 卷积核。
            # groups=in_channels 是实现深度卷积的关键。
            # padding=1 保证当 stride=1 时，特征图尺寸不变。
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels, # 深度卷积不改变通道数
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels, # 关键参数！
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.pointwise_conv = nn.Sequential(
            # --- 2. 逐点卷积 ---
            # 使用 1x1 卷积来组合深度卷积的输出通道。
            # 这允许我们改变输出的通道数。
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
            nn.Sigmoid()
        )

    def forward(self, x):
        """前向传播"""
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

@HEADS.register_module
class UTrackNetV1DWSHeadSigmoid(nn.Module):
    """
    它通过一个1x1卷积将输入特征图的通道数映射到任务所需的类别数。
    """
    def __init__(self, in_channels=64, out_channels=256):
        super().__init__()
        self.head = DepthwiseSeparableConvBlock(in_channels, out_channels)

    def forward(self, x):
        """
        输入: 来自Neck的精炼特征图, 形状为 [B, in_channels, H, W]
        输出: Logits, 形状为 [B, out_channels, H, W]
        """
        return self.head(x)

# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 1. 定义超参数和设备
    # 这些参数模拟了来自 Neck 模块的输出
    batch_size = 4
    in_channels = 64  # Neck 输出的通道数
    out_channels = 256 # 任务需要的最终输出通道数/类别数
    height = 640      # 特征图的高度
    width = 360       # 特征图的宽度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用的设备: {device}")

    # 2. 初始化 Head 网络
    # 使用我们定义的输入和输出通道数
    model = UTrackNetV1DWSHeadSigmoid(in_channels=in_channels, out_channels=out_channels).to(device)
    model.eval()

    # 3. 创建一个模拟的输入张量
    # 这个张量的形状应该和 Neck 模块的输出一致
    mock_input_tensor = torch.randn(batch_size, in_channels, height, width).to(device)
    print(f"\n模拟输入张量形状: {mock_input_tensor.shape}")

    # 4. 定义预期的输出形状
    # 1x1 卷积不改变 H 和 W, 只改变通道数
    expected_output_shape = (batch_size, out_channels, height, width)
    print(f"预期输出张量形状: {expected_output_shape}")

    print("\n--- 开始测试 ---")

    # 5. 执行前向传播
    with torch.no_grad():
        output_tensor = model(mock_input_tensor)

    # 6. 检查输出形状
    actual_output_shape = output_tensor.shape
    print(f"实际输出张量形状: {actual_output_shape}")

    # 7. 输出最终测试结果
    print("\n--- 测试结论 ---")
    if actual_output_shape == expected_output_shape:
        print("✅ 测试通过")
    else:
        print("❌ 测试不通过")