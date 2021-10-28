import torch
import torch.nn as nn
import torch.nn.functional as F


class ExistBCELoss(nn.Module):
    def __init__(self):
        super(ExistBCELoss, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.freq = torch.tensor([0.5, 0.5, 0.75, 0.25]).to(self.device)
        self.categories = ["ETT", "NGT", "CVC", "Swan Ganz Catheter Present"]
        self.loss_dict = {k: 0. for k in self.categories + ["Total"]}

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        freq = self.freq.expand_as(target)

        weight = torch.abs(target - freq)
        self.weight = weight

        total_loss = 0
        for i in range(len(self.categories)):
            self.loss_dict[self.categories[i]] = F.binary_cross_entropy(
                pred[:, i],
                target[:, i],
                weight[:, i],
            )
            total_loss += self.loss_dict[self.categories[i]]

        self.loss_dict["Total"] = total_loss / len(self.categories)

        return self.loss_dict["Total"]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = ExistBCELoss()
    output = torch.FloatTensor([[1, 50, 100, 1000]]).to(device)
    target = torch.FloatTensor([[1, 1, 0, 1]]).to(device)

    loss = criterion(output, target)
    print(loss)
    # print(criterion.loss_dict)
