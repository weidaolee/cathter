import abc
import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.loss_dict = {}
        self._factor = 0
        self.name = "BaseLoss"

    def _forward(self, output, target):
        """ current loss forward logic. """
        return 0

    def forward(self, output, target):
        """ Total loss recursion starting point. """
        total_loss = 0
        current_loss = self._forward(output, target)
        total_loss += current_loss * self._factor
        self.loss_dict["total"] = total_loss

        return total_loss


class BaseLossDecorator(BaseLoss, metaclass=abc.ABCMeta):
    def __init__(self, base_loss=BaseLoss(), factor=1):
        super(BaseLossDecorator, self).__init__()
        self.base_loss = base_loss
        self.loss_dict = base_loss.loss_dict
        self._factor = factor

    @abc.abstractmethod
    def _forward(self, output, target):
        """ This method sould be overwirted by its sub-class """
        ...

    def forward(self, output, target):
        """ Total loss recursion logic. """
        total_loss = self.base_loss.forward(output=output, target=target)
        current_loss = self._forward(output=output, target=target)

        total_loss += current_loss * self._factor

        self.loss_dict["total"] = total_loss
        return total_loss
