import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    二分类的 Focal Loss. 若要多分类需要再改。
    alpha:  对正样本的权重因子 (在不平衡任务中一般 <1 或 >1)
    gamma:  处理难/易样本的调节因子，默认是2
    reduction: 'none' | 'mean' | 'sum'
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: [batch_size, ...] 未经过 sigmoid
        targets: [batch_size, ...] 二进制标签(0/1)，和 logits 形状匹配
        """
        # 常规的 BCE with logits，每个样本的损失是 element-wise
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            reduction='none'
        )  # shape: same as logits

        # 这里 pt 是预测为标签那一方的概率
        # 注意对 logits 先做 sigmoid
        pred_prob = torch.sigmoid(logits)  
        # 针对正样本时，pt=pred_prob；针对负样本时，pt=1-pred_prob
        # 可以用下面的 trick 一步处理：pt = pred_prob * target + (1 - pred_prob) * (1 - target)
        pt = pred_prob * targets + (1 - pred_prob) * (1 - targets)
        
        # focal term: alpha * (1-pt)^gamma
        focal_term = self.alpha * (1 - pt).pow(self.gamma)

        loss = focal_term * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
