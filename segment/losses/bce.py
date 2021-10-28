import numpy as np

import torch
import torch.nn.functional as F

from losses.base import BaseLoss
from losses.base import BaseLossDecorator


class SegmentBCELoss(BaseLossDecorator):
    def __init__(
            self,
            base_loss=BaseLoss(),
            weight=None,
            size_average=False,
            factor=1,
    ):
        super(SegmentBCELoss, self).__init__(
            base_loss=base_loss,
            factor=factor,
        )
        self.name = "SegmentBCELoss"

    def _forward(self, output, target):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        output = output["seg"].to(device)
        target = target["seg"].to(device)

        loss = F.binary_cross_entropy(output, target, reduction='mean')
        self.loss_dict["seg_bce"] = loss
        return loss


class ClassifyBCELoss(BaseLossDecorator):
    def __init__(
            self,
            key,
            base_loss=BaseLoss(),
            weight=[],
            findings=[],
            factor=1,
    ):
        super(ClassifyBCELoss, self).__init__(
            base_loss=base_loss,
            factor=factor,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.key = key
        self.weight = torch.from_numpy(np.array(weight)).to(device)
        self.findings = findings
        self.name = "ClassifyBCELoss"

    def _forward(self, output, target):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        output = output["cls"][self.key].to(device)
        target = target["cls"][self.key].to(device)

        weight = self.weight.expand_as(target)
        weight = torch.abs(target - weight)

        total_loss = 0
        n = len(self.findings)
        for i in range(n):
            self.loss_dict[self.findings[i]] = F.binary_cross_entropy(
                output[:, i],
                target[:, i],
                weight[:, i],
            )
            total_loss += self.loss_dict[self.findings[i]] / n
        self.loss_dict[f"{self.key}_bce"] = total_loss
        return total_loss
