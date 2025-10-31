import torch
import torch.nn as nn
from ..builder import HEADS

# Motion fusion layers
class FusionLayerTypeA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        feature_map, attention_map = inputs
        output_1 = feature_map[:, 0, :, :] # 啥都不干
        output_2 = feature_map[:, 1, :, :] * attention_map[:, 0, :, :] # 只乘变亮通道
        output_3 = feature_map[:, 2, :, :] * attention_map[:, 3, :, :] # 只乘变亮通道

        return torch.stack([output_1, output_2, output_3], dim=1)


class R_STRHead(nn.Module):
    def __init__(self, 
                 in_channels=64,       # Backbone的输入通道
                 out_channels=3,       # 最终输出通道 (t-1, t, t+1)
                 img_size=(288, 512),  # "草稿"热力图的尺寸 (H, W)
                 patch_size=16,        # 分块大小
                 embed_dim=128,        # Transformer的工维度
                 num_transformer_layers=4, # Transformer层数
                 num_transformer_heads=8,   # Transformer多头注意力的头数
                 IsDraft=False # 是否需要返回草稿
                ):
        super().__init__()
        self.IsDraft = IsDraft
        self.fusion_layer = FusionLayerTypeA()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size= 3,  
                bias=False,  
                padding = 1
            ),
            nn.BatchNorm2d(out_channels), # 你可以根据需要添加
            nn.Sigmoid(),      # 你可以根据需要添加
        )
        self.embed_conv = nn.Conv2d(
            in_channels=1, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

        H_img, W_img = img_size
        H_feat, W_feat = H_img // patch_size, W_img // patch_size  # (18, 32)
        num_patches_per_frame = H_feat * W_feat                 # 576
        

        # (1) 空间位置编码 (在 t-1, t, t+1 之间共享)
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, num_patches_per_frame, embed_dim))
        # (2) 时间位置编码 (t-1, t, t+1)
        self.time_embed = nn.Parameter(torch.randn(1, 3, embed_dim)) # 3个时间步

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_transformer_heads, 
            batch_first=True,  # 确认输入格式为 [B, N, D]
            dim_feedforward=embed_dim * 4 # 标准配置
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_transformer_layers
        )
        
        self.decoder_head = nn.ConvTranspose2d(
            in_channels=embed_dim, 
            out_channels=1,        # 每次还原1个通道
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        self.final_sigmoid = nn.Sigmoid()


    def forward(self, x, residual_maps):
        draft = self.conv1(x)
        draft_after_mvdr_attention = self.fusion_layer([draft, residual_maps])

        # 保存 Batch_size 和 H, W 供后续使用
        B, C, H, W = draft_after_mvdr_attention.shape
        H_feat, W_feat = H // self.patch_size, W // self.patch_size # 18, 32
        # -----------------------------------------------
        # 2. 时空注意力模块
        # -----------------------------------------------
        
        # 2.1 拆分并添加通道维度
        # .unsqueeze(1) 将 [B, H, W] 变为 [B, 1, H, W]
        draft_prev = draft_after_mvdr_attention[:, 0, :, :].unsqueeze(1) # t-1
        draft_curr = draft_after_mvdr_attention[:, 1, :, :].unsqueeze(1) # t 
        draft_next = draft_after_mvdr_attention[:, 2, :, :].unsqueeze(1) # t+1

        # 2.2 各自分块嵌入
        embd_prev = self.embed_conv(draft_prev) # [B, 128, 18, 32]
        embd_curr = self.embed_conv(draft_curr) 
        embd_next = self.embed_conv(draft_next)

        # 2.3 展平并转置 (一步到位)
        # [B, 128, 18, 32] -> flatten(2) -> [B, 128, 576] -> permute(0,2,1) -> [B, 576, 128]
        flat_prev = embd_prev.flatten(2).permute(0, 2, 1)
        flat_curr = embd_curr.flatten(2).permute(0, 2, 1)
        flat_next = embd_next.flatten(2).permute(0, 2, 1)

        # 2.4 添加分解式位置编码 (你的方案)
        # 提取时间编码, 形状 [1, 1, 128] 以便广播
        time_prev_embed = self.time_embed[:, 0, :].unsqueeze(1)
        time_curr_embed = self.time_embed[:, 1, :].unsqueeze(1)
        time_next_embed = self.time_embed[:, 2, :].unsqueeze(1)
        
        # 总编码 = Patch数据 + 空间编码 + 时间编码
        in_prev = flat_prev + self.spatial_pos_embed + time_prev_embed
        in_curr = flat_curr + self.spatial_pos_embed + time_curr_embed
        in_next = flat_next + self.spatial_pos_embed + time_next_embed

        # 2.5 拼接成时空序列
        # [B, 1728, 128] (N = 576 * 3)
        final_input_with_pos = torch.cat([in_prev, in_curr, in_next], dim=1)
        
        # 2.6 叠Transformer编码器 ("精修")
        # repaired_sequence 形状仍然是 [B, 1728, 128]
        repaired_sequence = self.transformer_encoder(final_input_with_pos)

        # 2.7 解码
        
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

        # 2.8 用解码器头上采样
        # 【修改】: 输出的是 "修正值" (Residual)
        out_prev_residual = self.decoder_head(repaired_prev_feat) # [B, 1, 288, 512]
        out_curr_residual = self.decoder_head(repaired_curr_feat)
        out_next_residual = self.decoder_head(repaired_next_feat)

        # 3. 最终拼回 [B, 3, H, W] 并应用残差连接
        
        # [B, 3, 288, 512]
        residual = torch.cat([out_prev_residual, out_curr_residual, out_next_residual], dim=1)
        
        # 【【【【【 关键修改点 】】】】】
        # 将 "草稿" 和 "修正值" 相加
        final_output_before_sigmoid = draft_after_mvdr_attention + residual
        
        # 在相加之后，再进行 Sigmoid 激活
        final_output = self.final_sigmoid(final_output_before_sigmoid)
        if self.IsDraft:
            return final_output, draft_after_mvdr_attention
        else:
            return final_output