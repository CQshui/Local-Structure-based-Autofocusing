import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required
import torch.nn.functional as F


# loss function
def cross_entropy_loss_RCF(prediction, label, neg_boost=2.0):
    label = label.long()
    mask = label.float()

    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()

    total = num_positive + num_negative + 1e-6

    # 原始 class balance
    w_pos = num_negative / total
    w_neg = num_positive / total

    # 提高负样本惩罚
    w_neg = w_neg * neg_boost

    mask[mask == 1] = w_pos
    mask[mask == 0] = w_neg
    mask[mask == 2] = 0

    cost = F.binary_cross_entropy(
        prediction.float(),
        label.float(),
        weight=mask,
        reduction='none'
    )

    costs = F.relu(torch.sum(cost))
    return costs
