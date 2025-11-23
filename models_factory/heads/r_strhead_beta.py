import torch
import torch.nn as nn
from ..builder import HEADS

@HEADS.register_module
class R_STRHeadBeta(nn.Module):
    def __init__(self, 
                 in_channels=64,
                 out_channels=3,
                 img_size=(288, 512),
                 patch_size=16,
                 embed_dim=256,
                 num_transformer_layers=4,
                 num_transformer_heads=2, # 双流分工，双头配置
                 IsDraft=False,
                 mdd_channels=4 # MDD 4通道 (Bright/Dark * 2)
                ):
        super().__init__()
        self.IsDraft = IsDraft
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mdd_channels = mdd_channels

        # 1. 视觉草稿头
        self.draft_head = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
        # 2. Patch Embedding: 接受 Draft(1) + MDD(4) = 5 通道
        self.embed_conv = nn.Conv2d(
            in_channels=1 + mdd_channels, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

        # 位置编码
        H_img, W_img = img_size
        H_feat, W_feat = H_img // patch_size, W_img // patch_size 
        num_patches_per_frame = H_feat * W_feat 
        
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, num_patches_per_frame, embed_dim))
        self.time_embed = nn.Parameter(torch.randn(1, 3, embed_dim))

        # 核心组件: LN + Dropout
        self.input_norm = nn.LayerNorm(embed_dim)
        self.context_dropout = nn.Dropout(p=0.1)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_transformer_heads, 
            batch_first=True, 
            dim_feedforward=embed_dim*4, 
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # 解码头
        self.decoder_head = nn.Sequential(
            nn.Conv2d(embed_dim, 1 * (patch_size ** 2), kernel_size=1),
            nn.PixelShuffle(patch_size)
        )
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, x, mvdr_attention):
        # 1. 生成视觉草稿
        draft_logits = self.draft_head(x) 
        
        # 2. 【关键】: Contextual Forcing
        # draft_for_transformer 在训练时是"有毒"的 (被随机挖空)
        if self.training:
            draft_for_transformer = self.context_dropout(draft_logits)
        else:
            draft_for_transformer = draft_logits

        # 3. 准备 Transformer 输入 (双流拼接)
        d_prev = draft_for_transformer[:, 0, :, :].unsqueeze(1) 
        d_curr = draft_for_transformer[:, 1, :, :].unsqueeze(1)
        d_next = draft_for_transformer[:, 2, :, :].unsqueeze(1)
        
        # MDD 全量广播拼接
        mdd_map = mvdr_attention 

        input_prev = torch.cat([d_prev, mdd_map], dim=1) # [B, 5, H, W]
        input_curr = torch.cat([d_curr, mdd_map], dim=1)
        input_next = torch.cat([d_next, mdd_map], dim=1)

        # 4. Embedding + LayerNorm
        embd_prev = self.embed_conv(input_prev).flatten(2).permute(0, 2, 1)
        embd_curr = self.embed_conv(input_curr).flatten(2).permute(0, 2, 1)
        embd_next = self.embed_conv(input_next).flatten(2).permute(0, 2, 1)
        
        flat_prev = self.input_norm(embd_prev)
        flat_curr = self.input_norm(embd_curr)
        flat_next = self.input_norm(embd_next)

        # 5. Positional Encoding
        time_prev = self.time_embed[:, 0, :].unsqueeze(1)
        time_curr = self.time_embed[:, 1, :].unsqueeze(1)
        time_next = self.time_embed[:, 2, :].unsqueeze(1)
        
        in_prev = flat_prev + self.spatial_pos_embed + time_prev
        in_curr = flat_curr + self.spatial_pos_embed + time_curr
        in_next = flat_next + self.spatial_pos_embed + time_next

        # 6. Transformer
        final_input_seq = torch.cat([in_prev, in_curr, in_next], dim=1)
        repaired_seq = self.transformer_encoder(final_input_seq)

        # 7. Decode Residuals
        repaired_prev, repaired_curr, repaired_next = torch.chunk(repaired_seq, 3, dim=1)
        
        def decode_feat(flat_feat):
            B, N, D = flat_feat.shape
            H_feat, W_feat = int(self.img_size[0]/self.patch_size), int(self.img_size[1]/self.patch_size) # 动态获取或使用成员变量
            # 注意：这里为了稳健，最好直接用 draft 的 shape 推算 H_feat, W_feat
            # 你的代码里是在 forward 开头算的，这里可以直接用局部变量，或者简单重算
            feat_map = flat_feat.permute(0, 2, 1).reshape(B, self.embed_dim, H_feat, W_feat)
            return self.decoder_head(feat_map)
            
        # 这里的 H_feat, W_feat 需要确保在 forward 作用域内可见
        # 前面代码里有计算：H_feat, W_feat = H // self.patch_size, W // self.patch_size
        
        # 修正作用域问题，直接在这里计算
        B, _, H, W = draft_for_transformer.shape
        H_feat, W_feat = H // self.patch_size, W // self.patch_size
        
        out_prev_res = decode_feat(repaired_prev)
        out_curr_res = decode_feat(repaired_curr)
        out_next_res = decode_feat(repaired_next)
        
        residual_logits = torch.cat([out_prev_res, out_curr_res, out_next_res], dim=1)

        # 8. 【终极修改】: 有毒残差连接
        # 我们把 residual 加到 [draft_for_transformer] 上
        # 训练时：如果 draft 被 mask 了，residual 必须全额补上
        # 推理时：draft 是好的，residual 依然全额补上 -> 信号极强
        final_logits = draft_for_transformer + residual_logits
        
        if self.IsDraft:
             return self.final_sigmoid(final_logits), self.final_sigmoid(draft_logits)
        else:
             return self.final_sigmoid(final_logits)