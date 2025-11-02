import torch
import torch.nn as nn
from ..builder import BACKBONES, HEADS


class ConvBlock(nn.Module):
    """
    Simoidhead特质卷积块，符合wbce loss要求
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            # 2D卷积层
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,  # 3x3 的卷积核是标准选择
                bias=False  # 使用 BatchNorm 时，卷积层的偏置(bias)是多余的，可以省略
            ),
            nn.Sigmoid()  # 变为sigmoid用于wbce损失
        )

    def forward(self, x):
        return self.conv(x)


@HEADS.register_module
class TrackNetV2Head(nn.Module):
    """
    它通过一个1x1卷积将输入特征图的通道数映射到任务所需的类别数。
    """

    def __init__(self, in_channels=64, out_channels=3):
        super().__init__()
        self.head = ConvBlock(in_channels, out_channels)

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
    out_channels = 3  # 任务需要的最终输出通道数/类别数
    height = 640  # 特征图的高度
    width = 360  # 特征图的宽度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"使用的设备: {device}")

    # 2. 初始化 Head 网络
    # 使用我们定义的输入和输出通道数
    model = UTrackNetV2Head(in_channels=in_channels, out_channels=out_channels).to(device)
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