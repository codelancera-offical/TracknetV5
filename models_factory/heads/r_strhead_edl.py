import torch
import torch.nn as nn
from ..builder import HEADS


# FusionLayerTypeA 保持不变...
class FusionLayerTypeA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        feature_map, attention_map = inputs
        output_1 = feature_map[:, 0:2, :, :]
        output_2 = feature_map[:, 2:4, :, :] * attention_map[:, 0, :, :]
        output_3 = feature_map[:, 4:6, :, :] * attention_map[:, 2, :, :]
        return torch.stack([output_1, output_2, output_3], dim=1)


@HEADS.register_module
class R_STRHead_EDL(nn.Module):
    def __init__(self,
                 in_channels=64,
                 out_channels=3,
                 img_size=(288, 512),
                 patch_size=16,
                 embed_dim=256,
                 num_transformer_layers=4,
                 num_transformer_heads=1,
                 IsDraft=False
                 ):
        super().__init__()
        self.IsDraft = IsDraft
        self.fusion_layer = FusionLayerTypeA()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        print("head:", num_transformer_heads)
        # 1. 草稿头 (保持你之前的修改: 1x1 Conv)
        self.draft_head = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)

        # 2. Patch 嵌入层 (保持不变，我们坚持不引入 extra features)
        self.embed_conv = nn.Conv2d(
            in_channels=1,  # 坚持只看 1 通道概率图
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        H_img, W_img = img_size
        H_feat, W_feat = H_img // patch_size, W_img // patch_size
        num_patches_per_frame = H_feat * W_feat

        # Positional Embeddings (保持不变)
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, num_patches_per_frame, embed_dim))
        self.time_embed = nn.Parameter(torch.randn(1, 3, embed_dim))

        self.context_dropout = nn.Dropout(p=0.1)

        # Transformer Encoder (保持不变)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_transformer_heads,
            batch_first=True,
            dim_feedforward=embed_dim * 4,
            dropout=0.1  # 注意：Transformer 内部也有 dropout，保持默认即可
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # 解码头 (保持你之前的修改: PixelShuffle)
        self.decoder_head = nn.Sequential(
            nn.Conv2d(embed_dim, 1 * (patch_size ** 2), kernel_size=1),
            nn.PixelShuffle(patch_size)
        )

        self.final_ReLu = nn.ReLU()

    def forward(self, x, residual_maps):
        # 1. 生成草稿
        draft_logits = self.draft_head(x)

        # 2. 运动融合 (MVDR)
        # draft_clean 是我们最“干净”的草稿，用于最后的残差相加
        draft_clean = self.fusion_layer([draft_logits, residual_maps])
        B, C, K, H, W = draft_clean.shape

        # 把类比通道融入到b中
        draft_clean_reshape = draft_clean.permute(0, 2, 1, 3, 4).reshape(B*K, C, H, W) # [Bk, 3, H, W]
        Bk, C, H, W = draft_clean_reshape.shape

        if self.training:
            # 仅在训练时破坏输入，强迫模型“脑补”
            # 注意：我们只破坏送入 Transformer 的这份数据，
            # 最后的残差连接 (final_logits) 还是要加在 clean 的草稿上。
            draft_for_transformer = self.context_dropout(draft_clean_reshape)
        else:
            # 推理时不破坏，模型会利用全量信息进行更强的推理
            draft_for_transformer = draft_clean_reshape


        H_feat, W_feat = H // self.patch_size, W // self.patch_size

        # 3. 时空注意力流程

        # 3.1 拆分 (注意：这里用的是可能被 Dropout 过的 draft_for_transformer)
        draft_prev = draft_for_transformer[:, 0, :, :].unsqueeze(1)
        draft_curr = draft_for_transformer[:, 1, :, :].unsqueeze(1)
        draft_next = draft_for_transformer[:, 2, :, :].unsqueeze(1)

        # 3.2 Patch Embedding
        embd_prev = self.embed_conv(draft_prev)
        embd_curr = self.embed_conv(draft_curr)
        embd_next = self.embed_conv(draft_next)

        # 3.3 展平
        flat_prev = embd_prev.flatten(2).permute(0, 2, 1)  # [B, N, D]
        flat_curr = embd_curr.flatten(2).permute(0, 2, 1)
        flat_next = embd_next.flatten(2).permute(0, 2, 1)

        # ---------------------------------------------------------
        # 【关键修改 1 的应用】: LayerNorm 归一化
        # ---------------------------------------------------------
        # 在加上位置编码之前，先做归一化。
        # 这样，无论 Draft 里的概率是 0.1 还是 0.9，生成的 Token 向量模长都差不多。
        # Transformer 将被迫关注 "Pattern" (是不是球的形状/运动趋势) 而不是 "Intensity" (数值大小)。
        # flat_prev = self.input_norm(flat_prev)
        # flat_curr = self.input_norm(flat_curr)
        # flat_next = self.input_norm(flat_next)

        # 3.4 加位置编码 (保持不变)
        time_prev_embed = self.time_embed[:, 0, :].unsqueeze(1)
        time_curr_embed = self.time_embed[:, 1, :].unsqueeze(1)
        time_next_embed = self.time_embed[:, 2, :].unsqueeze(1)

        in_prev = flat_prev + self.spatial_pos_embed + time_prev_embed
        in_curr = flat_curr + self.spatial_pos_embed + time_curr_embed
        in_next = flat_next + self.spatial_pos_embed + time_next_embed

        # 3.5 拼接 & Transformer
        final_input_with_pos = torch.cat([in_prev, in_curr, in_next], dim=1)
        repaired_sequence = self.transformer_encoder(final_input_with_pos)

        # 3.6 & 3.7 解码回 Logits
        repaired_prev_flat, repaired_curr_flat, repaired_next_flat = torch.chunk(repaired_sequence, 3, dim=1)

        repaired_prev_feat = repaired_prev_flat.permute(0, 2, 1).reshape(Bk, self.embed_dim, H_feat, W_feat)
        repaired_curr_feat = repaired_curr_flat.permute(0, 2, 1).reshape(Bk, self.embed_dim, H_feat, W_feat)
        repaired_next_feat = repaired_next_flat.permute(0, 2, 1).reshape(Bk, self.embed_dim, H_feat, W_feat)

        out_prev_residual = self.decoder_head(repaired_prev_feat)
        out_curr_residual = self.decoder_head(repaired_curr_feat)
        out_next_residual = self.decoder_head(repaired_next_feat)

        # 4. 残差连接
        residual_logits = torch.cat([out_prev_residual, out_curr_residual, out_next_residual], dim=1)

        # 【注意】: 这里一定要加在 clean 的草稿上，不要加在 dropout 过的草稿上！
        # 我们希望：即使 Transformer 是看着残缺图算出的 Delta，也要把它加回到原始图上。
        final_logits = draft_for_transformer + residual_logits

        # 5. 最终激活
        final_output = self.final_ReLu(final_logits).view(B, -1, H, W)

        if self.IsDraft:
            return final_output, self.final_sigmoid(draft_for_transformer)
        else:
            return final_output