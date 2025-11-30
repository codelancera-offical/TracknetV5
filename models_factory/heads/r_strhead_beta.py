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
                 num_transformer_heads=2, 
                 IsDraft=False,
                 mdd_channels=4):
        super().__init__()
        self.IsDraft = IsDraft
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mdd_channels = mdd_channels
        
        # 【修复1：缺啥补啥】直接把 img_size 存下来
        self.img_size = img_size 
        
        # 【修复2：静态计算】直接在这里把特征图的高宽算死，Forward里直接用
        self.H_feat = img_size[0] // patch_size
        self.W_feat = img_size[1] // patch_size

        # 1. 视觉草稿头
        self.draft_head = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
        # 2. Patch Embedding
        self.embed_conv = nn.Conv2d(
            in_channels=1 + mdd_channels, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

        # 3. 位置编码
        num_patches_per_frame = self.H_feat * self.W_feat
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, num_patches_per_frame, embed_dim))
        self.time_embed = nn.Parameter(torch.randn(1, 3, embed_dim))

        # 4. LN + Dropout
        self.input_norm = nn.LayerNorm(embed_dim)
        self.context_dropout = nn.Dropout(p=0.1)

        # 5. Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_transformer_heads, 
            batch_first=True, 
            dim_feedforward=embed_dim*4, 
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # 6. 解码头
        self.decoder_head = nn.Sequential(
            nn.Conv2d(embed_dim, 1 * (patch_size ** 2), kernel_size=1),
            nn.PixelShuffle(patch_size)
        )
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, x, mvdr_attention):
        # 1. 生成视觉草稿
        draft_logits = self.draft_head(x) 
        
        # 2. 训练时施加 Dropout
        if self.training:
            draft_for_transformer = self.context_dropout(draft_logits)
        else:
            draft_for_transformer = draft_logits

        # 获取 Batch Size
        B = draft_for_transformer.shape[0]

        # 3. 准备拼接输入
        d_prev = draft_for_transformer[:, 0, :, :].unsqueeze(1) 
        d_curr = draft_for_transformer[:, 1, :, :].unsqueeze(1)
        d_next = draft_for_transformer[:, 2, :, :].unsqueeze(1)
        
        input_prev = torch.cat([d_prev, mvdr_attention], dim=1) 
        input_curr = torch.cat([d_curr, mvdr_attention], dim=1)
        input_next = torch.cat([d_next, mvdr_attention], dim=1)

        # 4. Embedding + LN
        embd_prev = self.embed_conv(input_prev).flatten(2).permute(0, 2, 1)
        embd_curr = self.embed_conv(input_curr).flatten(2).permute(0, 2, 1)
        embd_next = self.embed_conv(input_next).flatten(2).permute(0, 2, 1)
        
        flat_prev = self.input_norm(embd_prev)
        flat_curr = self.input_norm(embd_curr)
        flat_next = self.input_norm(embd_next)

        # 5. 加位置编码
        time_prev = self.time_embed[:, 0, :].unsqueeze(1)
        time_curr = self.time_embed[:, 1, :].unsqueeze(1)
        time_next = self.time_embed[:, 2, :].unsqueeze(1)
        
        in_prev = flat_prev + self.spatial_pos_embed + time_prev
        in_curr = flat_curr + self.spatial_pos_embed + time_curr
        in_next = flat_next + self.spatial_pos_embed + time_next

        # 6. Transformer 推理
        final_input_seq = torch.cat([in_prev, in_curr, in_next], dim=1)
        repaired_seq = self.transformer_encoder(final_input_seq)

        # 7. 解码残差
        repaired_prev, repaired_curr, repaired_next = torch.chunk(repaired_seq, 3, dim=1)
        
        # 【修复3：使用静态变量】直接用 self.H_feat 和 self.W_feat
        def decode_feat(flat_feat):
            feat_map = flat_feat.permute(0, 2, 1).reshape(B, self.embed_dim, self.H_feat, self.W_feat)
            return self.decoder_head(feat_map)

        out_prev_res = decode_feat(repaired_prev)
        out_curr_res = decode_feat(repaired_curr)
        out_next_res = decode_feat(repaired_next)
        
        residual_logits = torch.cat([out_prev_res, out_curr_res, out_next_res], dim=1)

        # 8. 有毒残差连接
        final_logits = draft_for_transformer + residual_logits
        
        if self.IsDraft:
             return self.final_sigmoid(final_logits), self.final_sigmoid(draft_logits)
        else:
             return self.final_sigmoid(final_logits)