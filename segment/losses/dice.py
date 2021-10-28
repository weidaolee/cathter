from losses.base import BaseLoss, BaseLossDecorator


class DiceLoss(BaseLossDecorator):
    def __init__(
            self,
            base_loss=BaseLoss(),
            smooth=1e-4,
            factor=1,
    ):
        super(DiceLoss, self).__init__(base_loss=base_loss, factor=factor)
        self.smooth = smooth
        self.name = "DiceLoss"

    def _forward(self, output, target):
        output = output["seg"].cuda()
        target = target["seg"].cuda()

        output = output.view(-1)
        target = target.view(-1)

        smooth = self.smooth

        intersection = (output * target).sum()
        dice = (2. * intersection + smooth) / (output.sum() + target.sum() +
                                               smooth)

        loss = 1 - dice
        self.loss_dict["dice"] = loss

        return loss
