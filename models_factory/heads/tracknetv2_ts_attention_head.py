import torch
import torch.nn as nn
from ..builder import BACKBONES, HEADS

# 你原来的 ConvBlock (假设它在这里)
class ConvBlock(nn.Module):
    """
    Simoidhead特质卷积块，符合wbce loss要求
    【建议】: 这里的Sigmoid最好去掉，Sigmoid应该在Head的最后输出时才使用。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1, 
                bias=False
            ),
            # nn.Sigmoid() # 建议在这里移除，加到Head的最后
        )

    def forward(self, x):
        return self.conv(x)


@HEADS.register_module
class TrackNetV2TSATTHead(nn.Module):
    """
    时空注意力头：
        先卷积到3个通道, 然后使用我们讨论的“方法B”(时空序列注意力)来进行调整！
        三通道尺寸的输出为 Bx3x288x512, 每个通道分别代表 t-1, t, t+1时刻的网球位置的热力图预测形式！
    """
    
    def __init__(self, 
                 in_channels=64,       # Backbone的输入通道
                 out_channels=3,       # 最终输出通道 (t-1, t, t+1)
                 embed_dim=128,        # Transformer的工维度
                 patch_size=16,        # 分块大小
                 num_transformer_layers=4, # Transformer层数
                 num_transformer_heads=8   # Transformer多头注意力的头数
                ):
        super().__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 1. 初始预测 (生成 "草稿")
        self.conv1 = ConvBlock(in_channels, out_channels)
        
        # 2. 时空注意力模块 (Spatio-Temporal Attention Module)
        
        # 2.1 分块嵌入 (Patch Embedding)
        # 注意 in_channels=1，因为我们会把三张图拆开分别送进去
        self.embed_conv = nn.Conv2d(
            in_channels=1, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # 2.2 时空位置编码
        # 我们的输入尺寸是 (288, 512)，分块大小 16x16
        H_feat, W_feat = 288 // patch_size, 512 // patch_size  # (18, 32)
        num_patches_per_frame = H_feat * W_feat                 # 576
        num_total_patches = num_patches_per_frame * 3           # 1728
        
        self.pos_embed = nn.Parameter(torch.randn(1, num_total_patches, embed_dim))
        
        # 2.3 Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_transformer_heads, 
            batch_first=True  # 确认输入格式为 [B, N, D]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_transformer_layers
        )
        
        # 2.4 解码器 (Decoder) - 将Token还原回热力图
        # 这是一个简单的 "Patch Upsampling" 解码器
        self.decoder_head = nn.ConvTranspose2d(
            in_channels=embed_dim, 
            out_channels=1,        # 每次还原1个通道
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # 3. 最终的激活函数
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x 初始形状: [B, 64, H, W]
        
        # 1. 生成“草稿”预测
        x = self.conv1(x)  # [B, 3, 288, 512]
        
        # 保存 Batch_size 和 H, W 供后续使用
        B, C, H, W = x.shape
        H_feat, W_feat = H // self.patch_size, W // self.patch_size # 18, 32
        num_patches_per_frame = H_feat * W_feat                   # 576

        # 2. 时空注意力模块
        
        # 2.1 拆分并添加通道维度 【修正点1】
        # .unsqueeze(1) 将 [B, H, W] 变为 [B, 1, H, W]
        prev_x = x[:, 0, :, :].unsqueeze(1) # t-1
        curr_x = x[:, 1, :, :].unsqueeze(1) # t 
        next_x = x[:, 2, :, :].unsqueeze(1) # t+1

        # 2.2 各自分块嵌入
        embd_prev = self.embed_conv(prev_x) # [B, 128, 18, 32]
        embd_curr = self.embed_conv(curr_x) 
        embd_next = self.embed_conv(next_x)

        # 2.3 各自展平 【修正点2】
        # start_dim=2 保证形状变为 [B, 128, 576]
        flat_prev = torch.flatten(embd_prev, start_dim=2) 
        flat_curr = torch.flatten(embd_curr, start_dim=2)
        flat_next = torch.flatten(embd_next, start_dim=2)

        # 2.4 拼接成时空序列
        # [B, 128, 576*3] -> [B, 128, 1728]
        final_input = torch.cat([flat_prev, flat_curr, flat_next], dim=2) 
        
        # 2.5 维度转置 【修正点3】
        # [B, 128, 1728] -> [B, 1728, 128] 以符合Transformer的 [B, N, D] 格式
        final_input_permuted = final_input.permute(0, 2, 1)

        # 2.6 添加位置编码
        final_input_with_pos = final_input_permuted + self.pos_embed

        # 2.7 叠Transformer编码器 (核心)
        # repaired_sequence 形状仍然是 [B, 1728, 128]
        repaired_sequence = self.transformer_encoder(final_input_with_pos)

        # 2.8 解码
        
        # 拆分回 T-1, T, T+1
        # 每一块的形状都是 [B, 576, 128]
        repaired_prev_flat, repaired_curr_flat, repaired_next_flat = torch.chunk(
            repaired_sequence, 3, dim=1
        )
        
        # 还原回特征图形状 (Un-Flatten + Reshape)
        # [B, 576, 128] -> [B, 128, 576] -> [B, 128, 18, 32]
        repaired_prev_feat = repaired_prev_flat.permute(0, 2, 1).reshape(B, self.embed_dim, H_feat, W_feat)
        repaired_curr_feat = repaired_curr_flat.permute(0, 2, 1).reshape(B, self.embed_dim, H_feat, W_feat)
        repaired_next_feat = repaired_next_flat.permute(0, 2, 1).reshape(B, self.embed_dim, H_feat, W_feat)

        # 2.9 用解码器头上采样
        # [B, 128, 18, 32] -> [B, 1, 288, 512]
        out_prev = self.decoder_head(repaired_prev_feat)
        out_curr = self.decoder_head(repaired_curr_feat)
        out_next = self.decoder_head(repaired_next_feat)

        # 3. 最终拼回 [B, 3, H, W] 并应用 Sigmoid
        final_output = torch.cat([out_prev, out_curr, out_next], dim=1)
        final_output = self.final_sigmoid(final_output)

        return final_output