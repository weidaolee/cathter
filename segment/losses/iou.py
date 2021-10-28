import torch
from losses.base import BaseLoss, BaseLossDecorator


class IoULoss(BaseLossDecorator):
    def __init__(
            self,
            base_loss=BaseLoss(),
            smooth=1e-4,
            factor=1,
    ):
        super(IoULoss, self).__init__(base_loss=base_loss, factor=factor)
        self.smooth = smooth
        self.name = "IoULoss"

    def _forward(self, output, target):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        output = output["seg"].to(device)
        target = target["seg"].to(device)

        output = output.view(-1)
        target = target.view(-1)

        smooth = self.smooth

        intersection = (output * target).sum()
        total = (output + target).sum()
        union = total - intersection

        iou = (intersection + smooth) / (union + smooth)

        loss = 1 - iou
        self.loss_dict["iou"] = loss
        return loss
