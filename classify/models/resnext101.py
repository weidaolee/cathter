import torch
import torchvision.models as models
import torch.nn as nn
from torchsummary import summary


class ResNext101(nn.Module):
    def __init__(self, in_channels=9, out_features=11, dropout=0.5):
        super(ResNext101, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.dropout = dropout

        model = models.resnext101_32x8d(pretrained=False, progress=True)

        model.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=[7, 7],
            stride=[2, 2],
            padding=[3, 3],
            bias=False,
        )

        model.fc = nn.Sequential(
            nn.Linear(2048, out_features, bias=True),
            # nn.Linear(2048, 512, bias=True),
            # nn.ReLU(),
            # nn.Dropout(p=dropout),
            # nn.Linear(512, 128, bias=True),
            # nn.ReLU(),
            # nn.Dropout(p=dropout),
            # nn.Linear(128, out_features, bias=True),
            nn.Sigmoid(),
        )

        self.model = nn.DataParallel(model)

    def forward(self, x):
        return self.model(x)

    def summary(self, device):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        summary(self.model.to(device), (self.in_channels, 512, 512))


if __name__ == "__main__":
    import os
    from losses import WeightedBCELoss

    os.environ['CUDA_VISIBLE_DEVICES'] = "4,5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNext101()
    model.summary(device)

    image = torch.randn([4, 9, 512, 512]).to(device)

    output = model(image)
    target = torch.rand([4, 11]).to(device)

    criterion = WeightedBCELoss(4)

    loss = criterion(output, target)
    print(loss)
