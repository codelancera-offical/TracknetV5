import torch
import torch.nn as nn
import torch.nn.functional as F

class WBCE_Loss_FromLogits(nn.Module):
    """
    一个实现了您提供的 "WBCE" (变种 Focal Loss, 伽马=2) 的 nn.Module。
    
    它接收 Logits (模型的原始输出，未经过 Sigmoid 激活) 以保证数值稳定性。
    
    公式: -sum [ (1-P)^2 * Y * log(P) + P^2 * (1-Y) * log(1-P) ]
    其中 P = sigmoid(logits)
    """
    def __init__(self, reduction='mean'):
        """
        Args:
            reduction (str): 指定损失的聚合方式。'mean' (默认) 或 'sum'。
        """
        super().__init__()
        self.reduction = reduction
        
    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): 模型的原始输出，形状为 (N, *)。
            targets (torch.Tensor): 真实标签 (0或1)，形状与 logits 相同。
        """
        # 1. 将 Logits 转换为概率 P
        # 使用 sigmoid 激活函数， P = σ(logits)
        logits = torch.squeeze(logits)
        print(f"logits shape in WBCE loss: {logits.shape}")
        prob = torch.sigmoid(logits)
        

        # 将 targets转换为 1.0-0.0之间
        targets = targets / 255.0
        
        # 2. 计算标准 BCE 损失的 Log 部分
        # 使用 F.binary_cross_entropy_with_logits 内部的数值稳定 Log 计算 (L = -Y*log(P) - (1-Y)*log(1-P))
        # reduction='none' 得到逐元素的损失 L_i
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 3. 计算调节因子 (Modulating Factor)
        
        # 3.1. 正样本 (Y=1) 的调节因子: (1 - P)^2
        # Y*bce_loss 提取出正样本的损失项， (1-P)是其权重
        pt = prob * targets + (1 - prob) * (1 - targets) # pt 是正确类别的预测概率 P_t

        # 对于正样本 (Y=1): 权重因子 = (1 - P)^2
        # 对于负样本 (Y=0): 权重因子 = P^2
        # 这就是 (1-pt)^2，因为 pt 就是当前样本的正确预测概率 P_t
        focal_weight = (1.0 - pt).pow(2)


        # 4. 应用调节因子
        # 加权后的损失 = 原始 BCE * 调节因子
        weighted_loss = focal_weight * bce_loss
        
        # 5. 聚合损失
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss # 返回逐元素的损失

if __name__ == "__main__":
    pass