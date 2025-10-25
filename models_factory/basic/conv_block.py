# 文件: models_factory/basic/conv.py

import torch.nn as nn

class BasicConvBlock(nn.Module):
    """
    一个更通用的卷积块，可以自定义激活函数。
    包含：Conv2d -> BatchNorm2d -> Activation
    """
    def __init__(self, in_channels, out_channels, activation='relu', k=3):
        super().__init__()
        
        # 根据传入的参数选择激活函数
        if activation == 'relu':
            self.activation_layer = nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            self.activation_layer = nn.Sigmoid()
        # 你未来还可以添加更多选项，例如 'leaky_relu'
        # elif activation == 'leaky_relu':
        #     self.activation_layer = nn.LeakyReLU(inplace=True)
        else:
            # 如果不提供或提供不支持的激活函数，则不使用激活层
            self.activation_layer = nn.Identity()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=k,
                padding=(k - 1) // 2,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            self.activation_layer 
        )

    def forward(self, x):
        return self.conv(x)

