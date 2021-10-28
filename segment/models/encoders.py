import segmentation_models_pytorch as smp
from models.bottlenecks import ResNestBottleneck


def get_resnest_encoder_class(norm_layer, act_layer):
    class ResNestEncoder(smp.encoders.timm_resnest.ResNestEncoder):
        def __init__(self,
                     out_channels,
                     depth=5,
                     norm_layer=norm_layer,
                     act_layer=act_layer,
                     **kwargs):
            super().__init__(
                out_channels=out_channels,
                depth=depth,
                norm_layer=norm_layer,
                act_layer=act_layer,
                **kwargs,
            )

        def load_state_dict(self, state_dict, **kwargs):
            super().load_state_dict(state_dict, strict=False, **kwargs)

    return ResNestEncoder


# smp.encoders.encoders["timm-resnest200e"]["encoder"] = ResNestEncoder
