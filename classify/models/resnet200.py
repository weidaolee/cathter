import os
import timm
import torch

import torch.nn as nn
from torchsummary import summary


class ResNet200D(nn.Module):
    def __init__(self, model_name='resnet200d', in_channels=9, out_dim=11):
        super(ResNet200D, self).__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.model = timm.create_model(model_name, pretrained=False)
        self.model.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels,
                      32,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1),
                      bias=False), self.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32,
                      32,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      bias=False), self.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32,
                      64,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      bias=False))

        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(n_features, out_dim),
        )

    def BatchNorm2d(self, num_features):
        return nn.BatchNorm2d(
            num_features,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )

    def summary(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        summary(self.to(device), (self.in_channels, 512, 512))

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)

        return output


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = timm.create_model("resnet200d", pretrained=False)

    model = ResNet200D()
    model.summary()
