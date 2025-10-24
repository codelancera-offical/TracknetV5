import torch
import torch.nn as nn
from ..builder import BACKBONES

# 导入 BasicConvBlock 和我们新的 CBAM 模块
from ..basic import BasicConvBlock as ConvBlock
from ..basic import CBAM


@BACKBONES.register_module
class UTrackNetV1BackboneCBAM(nn.Module):
    def __init__(self, in_channels=9):
        super().__init__()
        # --- Encoder Layers ---
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 64)
        self.cbam1 = CBAM(64) # <-- 新增CBAM层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)
        self.cbam2 = CBAM(128) # <-- 新增CBAM层
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)
        self.cbam3 = CBAM(256) # <-- 新增CBAM层
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)
        self.cbam4 = CBAM(512) # <-- 新增CBAM层 (bottleneck)

    def forward(self, x):
        features = {}
        
        # --- Encoder ---
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.cbam1(x) # <-- 在这里应用CBAM
        features['skip1'] = x
        
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.cbam2(x) # <-- 在这里应用CBAM
        features['skip2'] = x
        
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.cbam3(x) # <-- 在这里应用CBAM
        features['skip3'] = x
        
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.cbam4(x) # <-- 在这里应用CBAM
        features['bottleneck'] = x
        
        return features

# ... (测试代码部分保持不变) ...