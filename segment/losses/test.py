import torch
import torch.nn as nn

from losses.base import BaseLoss
from losses.dice import DiceLoss
from losses.focal import BinaryFocalLoss
from losses.iou import IoULoss
from losses.lovasz import LovaszLoss
from losses.bce import SegBCELoss, ClassBCELoss


class PostProcess(nn.Module):
    def __init__(self, max_pool=[1, 7]):
        super(PostProcess, self).__init__()
        self.max_pool = max_pool

    def forward(self, output, target):
        output = torch.sigmoid(output)
        output = self.dilation(output)
        target = self.dilation(target)

        return output, target

    def dilation(self, tensor):
        kernel_size = self.max_pool
        padding = [kernel_size[0] // 2, kernel_size[1] // 2]

        tensor = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=[1, 1],
            padding=padding,
            ceil_mode=True,
        )(tensor)
        return tensor


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight = [0.5, 0.5, 0.75, 0.25]
    findings = ["ETT", "NGT", "CVC", "SGC"]

    criterion = BaseLoss()
    criterion = DiceLoss(
        base_loss=criterion,
        factor=1,
    )
    criterion = BinaryFocalLoss(
        base_loss=criterion,
        gamma=2,
        factor=25.,
    )
    # criterion = SegBCELoss(
    #     base_loss=criterion,
    #     factor=1.,
    # )

    criterion = IoULoss(
        base_loss=criterion,
        factor=1.,
    )

    criterion = LovaszLoss(
        base_loss=criterion,
        classes=[1],
        # per_image=True,
        factor=1.)

    # criterion = ClassBCELoss(
    #     base_loss=criterion,
    #     weight=weight,
    #     findings=findings,
    #     factor=1,
    # )

    # image = plt.imread("target.jpg")[..., 0]
    image = np.ones([4, 512, 512])

    output = {"seg": None, "cls": None}
    target = {"seg": None, "cls": None}

    output["seg"] = torch.FloatTensor(image).to(device) * 100
    target["seg"] = torch.FloatTensor(np.ones_like(image)).to(device)

    output["seg"], target["seg"] = PostProcess()(output["seg"], target["seg"])

    output["cls"] = torch.FloatTensor([[1, 5, 7, 9]]).to(device)
    target["cls"] = torch.FloatTensor([[1, 1, 0, 1]]).to(device)

    loss = criterion(output, target)
    print(loss)
    print(criterion.loss_dict)

    # s = time.process_time()
    # loss = criterion(output, target)
    # print(time.process_time() - s)

    # print(loss)
    # print(criterion.loss_dict)
