import torch.nn as nn


def get_group_norm_class(num_groups):
    class GroupNorm(nn.GroupNorm):
        def __init__(
            self,
            num_channels,
            num_groups=num_groups,
            eps=1e-5,
            affine=True,
        ):
            super(GroupNorm, self).__init__(
                num_groups=num_groups,
                num_channels=num_channels,
                eps=eps,
                affine=affine,
            )

    return GroupNorm
