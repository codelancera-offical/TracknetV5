import torch
import torch.nn as nn

from ..builder import NECKS
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


@NECKS.register_module
class TrackNetV1Neck(nn.Module):
    def __init__(self):
        super().__init__()
        # ---Decoder Layers ---
        self.ups1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.tpconv1 = TripleConvBlock(512, 256)

        self.ups2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dbconv1 = DoubleConvBlock(256, 128)

        self.ups3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dbconv2 = DoubleConvBlock(128, 64)

    def forward(self, x):
        x = self.ups1(x)
        x = self.tpconv1(x)
        x = self.ups2(x)
        x = self.dbconv1(x)
        x = self.ups3(x)
        x = self.dbconv2(x)

        return x

if __name__ == "__main__":
    # 1. 定义超参数和设备
    # 这些参数应该与 Backbone 测试中的参数保持一致
    batch_size = 2
    in_channels = 512
    width = 80
    height = 45
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"使用的设备: {device}")

    # 2. 初始化 Neck 网络
    model = TrackNetV1Neck().to(device)
    model.eval()

    input_tensor = torch.randn(batch_size, in_channels, width, height).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    print(output_tensor.shape)