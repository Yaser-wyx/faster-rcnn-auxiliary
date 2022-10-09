import random

import numpy as np
from torch.nn.init import *

import logging

from torch import nn

DISTRIBUTION_LIST = [normal_, xavier_normal_, kaiming_normal_, xavier_uniform_, trunc_normal_, kaiming_uniform_,
                     orthogonal_, ]

logger = logging.getLogger("detectron2")


class RandomAuxiliaryInit(object):

    def __str__(self) -> str:
        return f"RandomAuxiliaryInit:\n" \
               f"distribution_num: {self.distribution_num}\n" \
               f"init_way: {self.init_way}\n" \
               f"distribution_fixed: {self.distribution_fixed}\n" \
               f"chosen_distributions: {self.chosen_distributions}\n"

    def __init__(self, distribution_num, init_way, distribution_fixed=False, fixed_distribution_list=None):
        if fixed_distribution_list is None:
            fixed_distribution_list = []
        self.distribution_num = distribution_num
        self.init_way = init_way
        self.distribution_fixed = distribution_fixed
        if len(fixed_distribution_list) > 0:
            global DISTRIBUTION_LIST
            DISTRIBUTION_LIST = []
            self.distribution_num = len(fixed_distribution_list)
            for distribution_init in fixed_distribution_list:
                DISTRIBUTION_LIST.append(globals()[distribution_init])
        self.chosen_distributions = self.get_random_distributions() if distribution_fixed else None
        self.init_func = None

    def get_random_distributions(self):
        choose_num = min(len(DISTRIBUTION_LIST), self.distribution_num)
        return np.random.choice(DISTRIBUTION_LIST, choose_num, replace=True)

    def random_choice_init_distribution_func(self):
        self.init_func = random.choice(self.chosen_distributions)

    def apply_random_init(self, model: nn.Module):

        if not self.distribution_fixed:
            self.chosen_distributions = self.get_random_distributions()

        if self.init_way == "whole-net":
            # in advance to get one distribution for initial
            self.random_choice_init_distribution_func()
        logger.info(str(_RANDOM_AUXILIARY_INIT))
        model.apply(random_weights_init)


_RANDOM_AUXILIARY_INIT: RandomAuxiliaryInit = None


def set_random_auxiliary_init(distribution_num, init_way, distribution_fixed=False, fixed_distribution_list=None):
    if fixed_distribution_list is None:
        fixed_distribution_list = []
    global _RANDOM_AUXILIARY_INIT
    _RANDOM_AUXILIARY_INIT = RandomAuxiliaryInit(distribution_num, init_way, distribution_fixed,
                                                 fixed_distribution_list)
    logger.info(str(_RANDOM_AUXILIARY_INIT))
    return _RANDOM_AUXILIARY_INIT


def get_random_auxiliary_init() -> RandomAuxiliaryInit:
    global _RANDOM_AUXILIARY_INIT
    assert _RANDOM_AUXILIARY_INIT is not None
    return _RANDOM_AUXILIARY_INIT


def random_weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        random_auxiliary_init = get_random_auxiliary_init()
        data = m.weight.data
        try:
            if random_auxiliary_init.init_way == "per-layer":
                random_auxiliary_init.random_choice_init_distribution_func()
            random_auxiliary_init.init_func(data)
        except:
            # it will run when data.dim() < 2 and random_auxiliary_init.init_func is not normal_ or uniform_
            init_func = random.choice([normal_, uniform_])
            init_func(data)
