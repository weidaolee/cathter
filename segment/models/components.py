import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None


class Dense(nn.Module):
    def __init__(self, in_features, out_features):
        super(Dense, self).__init__()
        self.dense = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.PReLU(num_parameters=in_features, init=0.25),
            nn.Linear(in_features=in_features, out_features=out_features),
        )

    def forward(self, x):
        return self.dense(x)


class Conv2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_norm=True,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU,
    ):
        super(Conv2D, self).__init__()

        self.norm_layer = norm_layer(in_channels)
        self.act_layer = act_layer()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_norm),
        )

        if use_norm == "inplace":
            self.act_layer = nn.Identity()

        elif use_norm and use_norm != "inplace":
            self.norm_layer = norm_layer(out_channels)

        else:
            self.norm_layer = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm_layer(x)
        x = self.act_layer(x)
        return x


class SCSEModule(nn.Module):
    def __init__(self, in_channels, act_layer, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            act_layer(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == 'scse':
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


class ClassificationHeadOutput(nn.Module):
    """
    Support dict multi-output for classification tasks.
    It means that you can do more than 2 classification tasks and
    1 segmentation task simultaneously.
    """
    def __init__(self, in_features, cls_out_features, dropout=0.5):
        super(ClassificationHeadOutput, self).__init__()
        self.in_features = in_features
        self.dropout = dropout
        self.cls_out_features = cls_out_features

        self.norm_layer = nn.BatchNorm1d(self.in_features)
        self.act_layer = nn.PReLU(self.in_features, init=0.25)

        self.linears = nn.ModuleDict({
            k: nn.Linear(
                in_features=in_features,
                out_features=cls_out_features[k],
            )
            for k in cls_out_features
        })

    def forward(self, x):
        x = self.norm_layer(x)
        # x = self.act_layer(x)
        x = {k: self.linears[k](x) for k in self.cls_out_features}
        return x


class CReLU(nn.Module):
    def __init__(self, inplace=True):
        super(CReLU, self).__init__()

    def forward(self, x):
        x = torch.cat([x, -x], dim=1)
        return F.relu(x)


class PReLU(nn.Module):
    def __init__(self, inplcae=False):
        super(PReLU, self).__init__()
        self.prelu = nn.PReLU(init=0.01)

    def forward(self, x):
        return self.prelu(x)
