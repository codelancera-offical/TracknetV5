import torch
import torch.nn as nn

from ..builder import LOSSES

def KL(alp, k, device):
    '''
    计算 Dir(alp) 和 Dir(1, ..., 1) 之间的KL散度。
    参数：
        alp: 证据向量，形状应为 [N, k]，N是批次大小或元素总数。
        k: 类别数
    '''
    beta = torch.ones([1, k], dtype=torch.float32, device=device)
    S_alpha = torch.sum(alp, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alp), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha) # [N, 1]
    dg1 = torch.digamma(alp)     # [N, 1]

    kl = torch.sum((alp - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni

    return kl

@LOSSES.register_module
class EDLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits:torch.tensor, targets: torch.Tensor, epoch_num=10, annealing_step=10, k=2) -> torch.tensor:
        '''
        参数；
            logits:模型输出的结果
            期望形状[B, 6, H, W]

            targets (torch.Tensor):
                   真实的灰度标签图 (Ground Truth Grayscale Map)。
                   期望形状: [B, 3, H, W]，其中每个像素的值是0或255
                   期望数据类型: torch.long (int64)。
        '''

        y = torch.where(targets == 255, 1.0, 0.0)
        B, ck, H, W = logits.shape
        C = ck / k

        # 转换矩阵[B, C, K, H, W] 得到evidence
        e = logits.view(B, -1, k, H, W) # [B, C, K, H, W]
        e1 = e[:, :, 0, :, :]  # 正样本的evidence
        e0 = e[:, :, 1, :, :]  # 负样本的evidence

        alpha1 = e1 + 1  # [b, c, H, W]
        alpha0 = e0 + 1

        S = alpha1 + alpha0

        p1= torch.div(alpha1, S)  # 正样本的belief
        p0 = torch.div(alpha0, S)  # 负样本的belief

        A = - (y * torch.log(p1) + (1 - y) * torch.log(p0))

        # 获得错误样本的one_hot编码
        y_one_hot = torch.stack([1.0 - y, y], dim=2) # [B, K, C, H, W]
        alp = e * y_one_hot + 1

        # 重塑为[N, K]形状，N = B * C * H * W
        N = int(B * C * H * W)
        alp_flat = alp.permute(0, 1, 3, 4, 2).reshape(N, k)  # [N, k]

        kl_flat = KL(alp_flat, k=k, device=logits.device)

        # 重塑或原始通道
        kl = kl_flat.view(B,-1, H, W)  # [B, C, H, W]

        # Annealing coefficient: 随着训练进行，逐渐增加KL项的权重
        lambda_t = min(1, epoch_num / annealing_step)
        B = lambda_t * kl

        loss = torch.mean((A + B))

        return loss


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("--- 测试 TrackNetV1Loss (基于256灰度等级分类原理) ---")

    # 定义模拟参数
    B, C, H, W = 4, 6, 640, 360  # 批大小=4, 类别/灰度等级=256, 尺寸=64x64

    # 1. 初始化损失函数
    loss_fn = EDLLoss()
    print("损失函数已初始化")

    # 2. 创建模拟输入
    # 模型的输出 logits, 形状 [B, 1, H, W]
    mock_logits = torch.randn(B, C, H, W)
    # 真实的灰度标签图, 形状 [B, H, W], 值为 0 到 255
    mock_targets = torch.randint(0, 255, (B, 3, H, W), dtype=torch.long)

    print(f"\n模拟 Logits 形状: {mock_logits.shape}")
    print(f"模拟 Targets 形状: {mock_targets.shape}")
    print(f"Targets 数据类型: {mock_targets.dtype}")

    # 3. 计算损失
    try:
        loss_value = loss_fn(torch.relu(mock_logits), mock_targets, 1)
        print(f"\n计算得到的损失值: {loss_value.item():.4f}")
        print(f"损失值是一个标量: {loss_value.dim() == 0}")
        print("\n✅ 测试通过：损失函数成功处理了符合原理的输入维度。")
    except Exception as e:
        print(f"\n❌ 测试失败. 错误: {e}")