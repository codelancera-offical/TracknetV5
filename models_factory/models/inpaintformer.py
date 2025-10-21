# model/inpaintformer.py

import torch
import torch.nn as nn
import math

from ..builder import MODELS

class PositionalEncoding(nn.Module):
    """
    为序列中的每个时间步添加位置信息。
    这是Transformer能够理解顺序的关键。
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

@MODELS.register_module
class InpaintFormer(nn.Module):
    def __init__(self, seq_len=60, embed_dim=128, num_heads=8, num_layers=6, dropout=0.1):
        super(InpaintFormer, self).__init__()

        self.embed_dim = embed_dim

        # 1. 输入嵌入层 将输入的(x, y, mask) 三维向量，映射到更高维度的 embed_dim空间，让其可表示更多信息
        self.input_embedding = nn.Linear(3, embed_dim)

        # 2. 位置编码
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)

        # 3. Transformer Encoder核心
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*4, # 128 -> 512 -> 128
            dropout=dropout, 
            batch_first=True # (N, L, C) N is batch size
        )

        # 把多个Encoder曾堆叠起来
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出头（Output Head）
        self.output_head = nn.Linear(embed_dim, 2) # 映射回(x,y)2维坐标

    def forward(self, coords, masks):
        # 输入形状: (N, L, C) -> N=批大小, L=序列长度, C=通道数
        # coords: (N, 60, 2), masks: (N, 60, 1)
        # masks为1代表可以信任该坐标，masks为0代表该坐标不可信，需要修复

        # 步骤1：准备输入
        # 将坐标和掩码拼接成(N, 60, 3)的张量
        x = torch.cat([coords, masks], dim=2)

        # 步骤2：输入嵌入
        x = self.input_embedding(x)

        # 步骤3：添加位置编码
        # 注意: PyTorch的PositionalEncoding默认接收 (L, N, C)，但我们的数据是 (N, L, C)
        # 所以我们需要转换一下维度
        x = x.premute(1, 0, 2)
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)

        # 步骤4： 通过Transformer Encoder
        transformer_output = self.transformer_encoder(x)

        # 步骤5：通过输出头获得最终坐标
        output_coords = self.output_head(transformer_output)

        # 步骤6：应用Sigmoid激活函数，确保输出在[0,1]之间
        return torch.sigmoid(output_coords)
