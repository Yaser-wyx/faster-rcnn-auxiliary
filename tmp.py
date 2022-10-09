from torch import nn

from detectron2.layers import Conv2d
from src.random_auxiliary_init import RandomAuxiliaryInit, set_random_auxiliary_init


class TMP(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size)


tmp = Conv2d(3, 32, kernel_size=3, )
random_auxiliary_init: RandomAuxiliaryInit = set_random_auxiliary_init(3, "per-layer")
random_auxiliary_init.apply_random_init(tmp)
