import torch
import torch.nn as nn

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
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """前向传播"""
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 1. 定义超参数和设备
    batch_size = 4
    in_channels = 13   # 输入通道数
    out_channels = 64 # 输出通道数
    height = 360       # 输入特征图的高度
    width = 640        # 输入特征图的宽度
    stride = 1         # 设置步长为2，将会使特征图尺寸减半
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    # 2. 初始化深度可分离卷积模型
    model = DepthwiseSeparableConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride
    ).to(device)
    model.eval()

    # 3. 创建一个模拟的输入张量
    mock_input_tensor = torch.randn(batch_size, in_channels, height, width).to(device)
    print(f"\n模拟输入张量形状: {mock_input_tensor.shape}")

    # 4. 定义预期的输出形状
    # H_out = (H_in - K + 2*P) / S + 1 = (128 - 3 + 2*1) / 2 + 1 = 64.5 -> 64
    # W_out = (W_in - K + 2*P) / S + 1 = (128 - 3 + 2*1) / 2 + 1 = 64.5 -> 64
    # 1x1 卷积不改变 H 和 W
    expected_output_shape = (batch_size, out_channels, height // stride, width // stride)
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
        
    # 8. 对比参数量
    # 标准卷积
    std_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
    std_params = sum(p.numel() for p in std_conv.parameters() if p.requires_grad)

    # 深度可分离卷积
    ds_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n--- 参数量对比 ---")
    print(f"标准卷积参数量: {std_params}")
    print(f"深度可分离卷积参数量: {ds_params}")
    print(f"参数量减少比例: {1 - ds_params / std_params:.2%}")