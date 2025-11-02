import torch
import torch.nn as nn
from ..builder import BACKBONES # 导入注册类，以供后续注册
from ..basic import BasicConvBlock as ConvBlock

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvBlock(self.in_channels, self.out_channels)
        self.conv2 = ConvBlock(self.out_channels, self.out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class TripleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvBlock(self.in_channels, self.out_channels)
        self.conv2 = ConvBlock(self.out_channels, self.out_channels)
        self.conv3 = ConvBlock(self.out_channels, self.out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

@BACKBONES.register_module
class TrackNetV1Backbone(nn.Module):
    def __init__(self, in_channels=9):
        super().__init__() # 调用父类初始化函数
        self.in_channels = in_channels
        # --- Encoder Layers ---
        self.dbconv1 = DoubleConvBlock(self.in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dbconv2 = DoubleConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.tpconv1 = TripleConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.tpconv2 = TripleConvBlock(256, 512)
    
    def forward(self, x):
        x = self.dbconv1(x)

        x = self.pool1(x)
        x = self.dbconv2(x)

        x = self.pool2(x)
        x = self.tpconv1(x)

        x = self.pool3(x)
        x = self.tpconv2(x)

        return x

if __name__ == "__main__":
    batch_size = 2
    input_height = 640
    input_width = 360
    in_channels = 9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用的设备: {device}")

    model = TrackNetV1Backbone(in_channels=in_channels).to(device)
    model.eval()

    test_tensor = torch.randn(batch_size, in_channels, input_height, input_width).to(device)
    print(f"输入张量形状: {test_tensor.shape}")

    # 5. 执行前向传播并获取输出
    with torch.no_grad(): # 在测试阶段不计算梯度
        output_features = model(test_tensor)
        print(output_features.shape)

