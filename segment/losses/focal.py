import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.base import BaseLoss, BaseLossDecorator


class BinaryFocalLoss(BaseLossDecorator):
    """
    This is a implementation of Focal Loss with smooth label cross entropy
    supported which is proposed in' Focal Loss for Dense Object Detection.
    (https://arxiv.org/abs/1708.02002)' Focal_Loss= -1 * alpha * (1-pt) * log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for
        well-classified examples (p>0.5) putting more focus on hard misclassified
        example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """
    def __init__(
            self,
            base_loss=BaseLoss(),
            gamma=2,
            factor=1,
    ):
        super(BinaryFocalLoss, self).__init__(base_loss=base_loss,
                                              factor=factor)

        self.gamma = gamma
        self.name = "FocalLoss"

    def _forward(self, output, target):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        output = output["seg"].to(device)
        target = target["seg"].to(device)

        bce_loss = F.binary_cross_entropy(output, target, reduction="mean")
        pt = torch.exp(-bce_loss)
        loss = ((1 - pt)**self.gamma * bce_loss).mean()

        self.loss_dict["focal"] = loss
        return loss
