import torch.nn as nn


class PReLUStrategy:
    def __init__(self, model, encoder_init, decoder_init):
        self.model = model
        self.encoder = model.encoder
        self.decoder = model.decoder

        self.encoder_init = encoder_init
        self.decoder_init = decoder_init

    def prepare_encoder(self):
        e = self.encoder
        c = e.conv1
        init = self.encoder_init
        c[2] = nn.PReLU(c[1].num_features, init=init)
        c[5] = nn.PReLU(c[4].num_features, init=init)

        for layer in [e.layer1, e.layer2, e.layer3, e.layer4]:
            for i in range(len(layer)):
                l = layer[i]
                c = l.conv2
                c.act0 = nn.PReLU(c.bn0.num_features, init=init)
                c.act1 = nn.PReLU(c.bn1.num_features, init=init)
                l.act1 = nn.PReLU(l.bn1.num_features, init=init)
                l.act3 = nn.PReLU(l.bn3.num_features, init=init)

    def prepare_decoder(self):
        init = self.decoder_init
        for b in self.decoder.blocks:
            for c in [b.conv1, b.conv2]:
                c.act_layer = nn.PReLU(c.norm_layer.num_features, init=0.001)

            for a in [b.attention1.attention.cSE, b.attention2.attention.cSE]:
                a[2] = nn.PReLU(a[1].out_channels, init=0.01)


class RReLUStrategy:
    def __init__(self, model, lower=1 / 8, upper=1 / 3):
        self.model = model
        self.encoder = model.encoder
        self.decoder = model.decoder

        self.lower = lower
        self.upper = upper

    def prepare_encoder(self):
        lower = self.lower
        upper = self.upper

        e = self.encoder
        c = e.conv1
        c[2] = nn.RReLU(lower, upper, inplace=c[2].inplace)
        c[5] = nn.PReLU(lower, upper, inplace=c[5].inplace)

        for layer in [e.layer1, e.layer2, e.layer3, e.layer4]:
            for i in range(len(layer)):
                l = layer[i]
                c = l.conv2
                c.act0 = nn.RReLU(lower, upper, inplace=c.act0.inplace)
                c.act1 = nn.RReLU(lower, upper, inplace=c.act1.inplace)
                l.act1 = nn.RReLU(lower, upper, inplace=l.act1.inplace)
                l.act3 = nn.RReLU(lower, upper, inplace=l.act3.inplace)

    def prepare_decoder(self):
        lower = self.lower
        upper = self.upper

        for b in self.decoder.blocks:
            for c in [b.conv1, b.conv2]:
                c.act_layer = nn.RReLU(lower,
                                       upper,
                                       inplace=c.act_layer.inplace)

            for a in [b.attention1.attention.cSE, b.attention2.attention.cSE]:
                a[2] = nn.RReLU(lower, upper, inplace=a[2].inplace)
