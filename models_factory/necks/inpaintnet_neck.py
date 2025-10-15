import torch
import torch.nn as nn
from ..builder import NECKS

# ==================== 建筑模块 ====================
class Conv1DBlock(nn.Module):
    """ Conv1D + LeakyReLU"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding='same', bias=True)
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

# ==================== Neck 主体 ====================
@NECKS.register_module
class InpaintNetNeck(nn.Module):
    """
    InpaintNet 的 Neck 部分 (1D 解码器)
    融合来自 Backbone 的多尺度 1D 特征。
    """
    def __init__(self):
        super().__init__()
        self.up_1 = Conv1DBlock(256 + 128, 128) # 输入: bottleneck(256) + skip3(128)
        self.up_2 = Conv1DBlock(128 + 64, 64)   # 输入: 上一层(128) + skip2(64)
        self.up_3 = Conv1DBlock(64 + 32, 32)    # 输入: 上一层(64) + skip1(32)

    def forward(self, features):
        """
        输入: 来自Backbone的特征字典
        输出: 精炼后的 1D 特征序列
        """
        x3 = features['skip3']
        x2 = features['skip2']
        x1 = features['skip1']
        x = features['bottleneck']

        x = torch.cat([x, x3], dim=1)
        x = self.up_1(x)
        
        x = torch.cat([x, x2], dim=1)
        x = self.up_2(x)
        
        x = torch.cat([x, x1], dim=1)
        x = self.up_3(x)
        
        return x

# ==================== 测试代码 ====================
if __name__ == "__main__":
    batch_size = 4
    sequence_length = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"使用的设备: {device}")

    model = InpaintNetNeck().to(device)
    model.eval()

    mock_features = {
        'skip1': torch.randn(batch_size, 32, sequence_length).to(device),
        'skip2': torch.randn(batch_size, 64, sequence_length).to(device),
        'skip3': torch.randn(batch_size, 128, sequence_length).to(device),
        'bottleneck': torch.randn(batch_size, 256, sequence_length).to(device)
    }
    
    print("\n--- 模拟输入特征形状 ---")
    for name, tensor in mock_features.items():
        print(f"  - {name}: {tensor.shape}")

    expected_output_shape = (batch_size, 32, sequence_length)
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