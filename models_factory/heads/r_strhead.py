import torch
import torch.nn as nn
from ..builder import HEADS

# 这个融合层是你的业务逻辑，我们保持不变。
# 评论：在Logit空间进行乘法门控 (Gating) 是一个非常好的操作。
class FusionLayerTypeA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        # -----------------------------------------------------------------
        # 【重要】: 
        # 这里的 feature_map (即 draft_logits) 现在是 Logit 空间的值 (例如 [-5, 5])
        # 你的 attention_map (residual_maps) 应该是一个 [0, 1] 的概率图
        # Logit * 概率 = 门控Logit (Gated Logit)。这是完美且正确的。
        # -----------------------------------------------------------------
        feature_map, attention_map = inputs
        output_1 = feature_map[:, 0, :, :] # 啥都不干
        output_2 = feature_map[:, 1, :, :] * attention_map[:, 0, :, :] # 门控 Logit
        output_3 = feature_map[:, 2, :, :] * attention_map[:, 2, :, :] # 门控 Logit

        return torch.stack([output_1, output_2, output_3], dim=1)

@HEADS.register_module
class R_STRHead(nn.Module):
    def __init__(self, 
                 in_channels=64,      # 来自Decoder/FPN的特征通道数
                 out_channels=3,      # 最终输出通道 (t-1, t, t+1)
                 img_size=(288, 512),   # "草稿"热力图的尺寸 (H, W)
                 patch_size=16,       # 分块大小
                 embed_dim=256,       # Transformer的工维度
                 num_transformer_layers=4, # Transformer层数
                 num_transformer_heads=1,  # Transformer多头注意力的头数
                 IsDraft=False # 是否需要返回草稿
                ):
        super().__init__()
        self.IsDraft = IsDraft
        self.fusion_layer = FusionLayerTypeA()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # -----------------------------------------------------------------
        # 【【【 关键修改 1: "草稿头" (Draft Head) 】】】
        # -----------------------------------------------------------------
        # 1. 移除重量级的 3x3 卷积 (之前是 256->64->3)
        # 2. 移除中间的 ReLU (它会丢失所有负面证据)
        # 3. 移除中间的 Sigmoid (它会压缩信息并阻止 Logit 修正)
        # 4. 替换为：单一、轻量、高效的 1x1 卷积。
        #    它是一个纯线性的 "逐像素解释器"。
        # -----------------------------------------------------------------
        self.draft_head = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1, # <--- 最佳实践: 1x1 
            padding=0,     # <--- 1x1 卷积不需要 padding
        )
        
        # -----------------------------------------------------------------
        # Patch 嵌入层 (保持不变, 卷积式嵌入是很好的做法)
        self.embed_conv = nn.Conv2d(
            in_channels=1, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

        H_img, W_img = img_size
        H_feat, W_feat = H_img // patch_size, W_img // patch_size  # (18, 32)
        num_patches_per_frame = H_feat * W_feat         # 576
        
        # (1) 空间位置编码 (保持不变)
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, num_patches_per_frame, embed_dim))
        # (2) 时间位置编码 (保持不变)
        self.time_embed = nn.Parameter(torch.randn(1, 3, embed_dim)) # 3个时间步

        # Transformer 编码器 (保持不变, 标准实现)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_transformer_heads, 
            batch_first=True,  # 确认输入格式为 [B, N, D]
            dim_feedforward=embed_dim * 4 
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_transformer_layers
        )
        
        # -----------------------------------------------------------------
        # 【【【 关键修改 2: "解码头" (Decoder Head) 】】】
        # -----------------------------------------------------------------
        # 1. 你的 ConvTranspose2d (转置卷积) 是可行的, 但容易产生棋盘格伪影。
        # 2. 替换为：更现代、更稳定的 PixelShuffle (像素重排) 上采样。
        #    它与 ConvTranspose2d 参数量相同, 但效果通常更好。
        #    它首先在低分辨率上用 1x1 卷积计算, 然后智能地 "重排" 像素到高分辨率。
        # -----------------------------------------------------------------
        self.decoder_head = nn.Sequential(
            nn.Conv2d(
                in_channels=embed_dim, 
                out_channels=1 * (patch_size ** 2), # 1个输出通道 * (放大倍数^2)
                kernel_size=1 # 1x1 卷积效率最高
            ),
            nn.PixelShuffle(patch_size) # 智能上采样
        )
        
        # -----------------------------------------------------------------
        # 【【【 关键修改 3: "最终激活" 】】】
        # -----------------------------------------------------------------
        # 整个流程中唯一的 Sigmoid, 放在所有计算 (包括残差相加) 之后。
        # -----------------------------------------------------------------
        self.final_sigmoid = nn.Sigmoid()


    def forward(self, x, residual_maps):
        # 1. 生成 "草稿 Logits"
        # x 是 [B, 64, H, W]
        # draft_logits 是 [B, 3, H, W], 值域是 (-inf, +inf)
        draft_logits = self.draft_head(x)
        
        # 2. 【你的业务逻辑】应用运动融合 (在 Logit 空间进行)
        # draft_after_mvdr_attention 仍然是 [B, 3, H, W] 的 Logits
        draft_after_mvdr_attention = self.fusion_layer([draft_logits, residual_maps])

        # 保存 Batch_size 和 H, W 供后续使用
        B, C, H, W = draft_after_mvdr_attention.shape
        H_feat, W_feat = H // self.patch_size, W // self.patch_size # 18, 32
        
        # -----------------------------------------------
        # 3. 时空注意力模块 (这部分逻辑不变)
        # -----------------------------------------------
        
        # 3.1 拆分并添加通道维度
        draft_prev = draft_after_mvdr_attention[:, 0, :, :].unsqueeze(1) # t-1
        draft_curr = draft_after_mvdr_attention[:, 1, :, :].unsqueeze(1) # t 
        draft_next = draft_after_mvdr_attention[:, 2, :, :].unsqueeze(1) # t+1

        # 3.2 各自分块嵌入 (Patch Embedding)
        embd_prev = self.embed_conv(draft_prev) # [B, 128, 18, 32]
        embd_curr = self.embed_conv(draft_curr) 
        embd_next = self.embed_conv(draft_next)

        # 3.3 展平并转置
        flat_prev = embd_prev.flatten(2).permute(0, 2, 1) # [B, 576, 128]
        flat_curr = embd_curr.flatten(2).permute(0, 2, 1)
        flat_next = embd_next.flatten(2).permute(0, 2, 1)

        # 3.4 添加分解式位置编码
        time_prev_embed = self.time_embed[:, 0, :].unsqueeze(1)
        time_curr_embed = self.time_embed[:, 1, :].unsqueeze(1)
        time_next_embed = self.time_embed[:, 2, :].unsqueeze(1)
        
        in_prev = flat_prev + self.spatial_pos_embed + time_prev_embed
        in_curr = flat_curr + self.spatial_pos_embed + time_curr_embed
        in_next = flat_next + self.spatial_pos_embed + time_next_embed

        # 3.5 拼接成时空序列
        final_input_with_pos = torch.cat([in_prev, in_curr, in_next], dim=1)
        
        # 3.6 叠Transformer编码器 ("精修")
        repaired_sequence = self.transformer_encoder(final_input_with_pos)

        # 3.7 解码
        
        # 拆分回 T-1, T, T+1
        repaired_prev_flat, repaired_curr_flat, repaired_next_flat = torch.chunk(
            repaired_sequence, 3, dim=1
        )
        
        # 还原回特征图形状
        repaired_prev_feat = repaired_prev_flat.permute(0, 2, 1).reshape(B, self.embed_dim, H_feat, W_feat)
        repaired_curr_feat = repaired_curr_flat.permute(0, 2, 1).reshape(B, self.embed_dim, H_feat, W_feat)
        repaired_next_feat = repaired_next_flat.permute(0, 2, 1).reshape(B, self.embed_dim, H_feat, W_feat)

        # 3.8 用解码器头上采样
        # 【重要】: 输出的是 "修正 Logits" (Residual Logits)
        # 它们的值域是 (-inf, +inf)
        out_prev_residual = self.decoder_head(repaired_prev_feat) # [B, 1, 288, 512]
        out_curr_residual = self.decoder_head(repaired_curr_feat)
        out_next_residual = self.decoder_head(repaired_next_feat)

        # -----------------------------------------------
        # 4. 最终拼回并应用残差连接 (在 Logit 空间)
        # -----------------------------------------------
        
        # [B, 3, 288, 512]
        residual_logits = torch.cat([out_prev_residual, out_curr_residual, out_next_residual], dim=1)
        
        # 【【【【【 核心点 】】】】】
        # (草稿 Logits) + (修正 Logits) = 最终 Logits
        final_logits = draft_after_mvdr_attention + residual_logits
        
        # -----------------------------------------------
        # 5. 最终激活
        # -----------------------------------------------
        # 在所有计算完成后，进行唯一一次 Sigmoid 激活, 得到最终概率
        final_output = self.final_sigmoid(final_logits)
        
        if self.IsDraft:
            # 【修改】: 如果需要返回草稿(例如用于辅助损失或可视化)
            # 我们也应该返回概率图, 所以在这里对它单独激活
            return final_output, self.final_sigmoid(draft_after_mvdr_attention)
        else:
            return final_output