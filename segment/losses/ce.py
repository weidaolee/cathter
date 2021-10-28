import numpy as np

import torch
import torch.nn.functional as F

from losses.base import BaseLoss
from losses.base import BaseLossDecorator


class CrossEntropyLoss(BaseLossDecorator):
    def __init__(
            self,
            key,
            base_loss=BaseLoss(),
            weight=[],
            factor=1,
    ):
        super(CrossEntropyLoss, self).__init__(
            base_loss=base_loss,
            factor=factor,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.key = key
        weight = torch.Tensor(weight)
        # weight = weight / weight.sum()
        weight = (1 - weight).to(device)
        self.weight = weight
        self.name = "ClassifyBCELoss"

    def _forward(self, output, target):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        output = output["cls"][self.key].to(device)
        target = target["cls"][self.key].to(device)

        output = torch.log(output)
        target = torch.argmax(target, dim=1)

        loss = F.nll_loss(output, target, weight=self.weight)

        self.loss_dict[f"{self.key}_ce"] = loss
        return loss
