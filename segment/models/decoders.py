import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components import Conv2D, Attention


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_norm,
        norm_layer,
        act_layer,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2D(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_norm,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        self.attention1 = Attention(
            name=attention_type,
            act_layer=act_layer,
            in_channels=in_channels + skip_channels,
        )
        self.conv2 = Conv2D(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_norm,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        self.attention2 = Attention(
            name=attention_type,
            act_layer=act_layer,
            in_channels=out_channels,
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_norm,
        norm_layer,
        act_layer,
    ):
        conv1 = Conv2D(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_norm,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        conv2 = Conv2D(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_norm,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        use_norm,
        norm_layer,
        act_layer,
        n_blocks=5,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks."
                .format(n_blocks, len(decoder_channels)))

        encoder_channels = encoder_channels[
            1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::
                                            -1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels,
                head_channels,
                use_norm=use_norm,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(
            use_norm=use_norm,
            norm_layer=norm_layer,
            act_layer=act_layer,
            attention_type=attention_type,
        )
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(
                in_channels,
                skip_channels,
                out_channels,
            )
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        # remove first skip with same spatial resolutio
        features = features[1:]
        # reverse channels to start from head of encoder
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
