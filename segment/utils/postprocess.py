import torch
import torch.nn as nn


class DilateSigmoidPostProcess(nn.Module):
    def __init__(self, max_pool=[1, 7]):
        super(DilateSigmoidPostProcess, self).__init__()
        self.max_pool = max_pool

    def forward(self, output, target):
        output, target = SigmoidPostProcess()(output, target)

        if "seg" in target:
            output["seg"] = self.dilation(output["seg"])
            target["seg"] = self.dilation(target["seg"])

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


class DilateSoftmaxPostProcess(nn.Module):
    def __init__(self, max_pool=[1, 7]):
        super(DilateSoftmaxPostProcess, self).__init__()
        self.max_pool = max_pool

    def forward(self, output, target):
        output, target = SoftmaxPostProcess()(output, target)

        if "seg" in target:
            output["seg"] = self.dilation(output["seg"])
            target["seg"] = self.dilation(target["seg"])

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


class AppearenceCorrectionPostProcess(nn.Module):
    def __init__(self):
        super(AppearenceCorrectionPostProcess, self).__init__()

    def forward(self, output, target):
        output, target = SigmoidPostProcess()(output, target)

        appear = (output["cls"]["appear"] > 0.5).float()
        status = output["cls"]["status"].clone()

        # mul = torch.ones_like(status)

        status[:, 0:3] = status[:, 0:3] * appear[:, 0:1]  # cvc
        status[:, 3:7] = status[:, 3:7] * appear[:, 1:2]  # ngt
        status[:, 7:10] = status[:, 7:10] * appear[:, 2:3]  # cvc
        # note: appear's size is [b, 4], appear[:, 0]'s size is [b], not [b, 1]

        # output["cls"]["status"] = status
        # output["cls"]["appear"] = appear

        output["cls"]["status"] = status

        return output, target


class SigmoidPostProcess(nn.Module):
    def __init__(self):
        super(SigmoidPostProcess, self).__init__()

    def forward(self, output, target):
        if "cls" in output:
            for k in output["cls"]:
                output["cls"][k] = torch.sigmoid(output["cls"][k])

        return output, target


class SoftmaxPostProcess(nn.Module):
    def __init__(self):
        super(SoftmaxPostProcess, self).__init__()

    def forward(self, output, target):
        for k in output["cls"]:
            if k in {"ett", "ngt"}:
                output["cls"][k] = torch.softmax(output["cls"][k], dim=1)

        return output, target
