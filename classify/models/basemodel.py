import torch
from torchsummary import summary


class SummaryModel:
    def summary(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        summary(self.to(device), (self.in_channels, 512, 512))
