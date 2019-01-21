# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018-09-21
"""
from model import *
from settings import *


if __name__ == '__main__':

    simulate(param={
        DATA_SET: ALL,
        BATCH_SIZE: 10000,
        EPOCHS: 100,
        ACTIVATION: RELU,
        BIAS_INITIALIZER: HE_UNIFORM,
        KERNEL_INITIALIZER: GLOROT_UNIFORM,
        BIAS_REGULARIZER: NONE,
        HIDDEN_LAYER: DNN8_4,
        DROPOUT: False,
        DROPOUT_RATE: 0.5
    }, case_number=9)
