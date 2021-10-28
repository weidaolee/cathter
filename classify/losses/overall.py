import torch
import torch.nn as nn
import torch.nn.functional as F


class OverallBCELoss(nn.Module):
    def __init__(self):
        super(OverallBCELoss, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.freq = torch.tensor([
            0.09996133 * 1.25,
            0.16854454,
            0.20569808,
            0.08781746 * 1.25,
            0.09604228 * 1.25,
            0.09111779 * 1.25,
            0.21371664,
            0.13064329 * 1.25,
            0.38365347 / 1.5,
            0.49544927 / 1.5,
            0.08578059 * 1.25,
        ]).to(self.device)

        self.categories = [
            'ETT - Abnormal',
            'ETT - Borderline',
            'ETT - Normal',
            'NGT - Abnormal',
            'NGT - Borderline',
            'NGT - Incompletely Imaged',
            'NGT - Normal',
            'CVC - Abnormal',
            'CVC - Borderline',
            'CVC - Normal',
            'Swan Ganz Catheter Present',
        ]
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
                # weight[:, i],
            )
            total_loss += self.loss_dict[self.categories[i]]

        self.loss_dict["Total"] = total_loss / len(self.categories)

        return self.loss_dict["Total"]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = OverallBCELoss()
    target = torch.FloatTensor([[1, 0, 1, .5, 0, 5, 0, 0, 1, 0]]).to(device)
    output = torch.FloatTensor([[1, 0, 0, 1., 0, 0, 0, 0, 0, 1, 0]]).to(device)

    loss = criterion(output, target)
    print(loss)
    print(criterion.loss_dict)
