# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018-09-21
"""
import gc

from model import *
from settings import *
from ensemble import INTERSECTION, GEOMETRIC

if __name__ == '__main__':
    get_forward_predict(param={
        DATA_SET: ALL,
        BATCH_SIZE: 300,
        EPOCHS: 1,
        ACTIVATION: LINEAR,
        BIAS_INITIALIZER: HE_UNIFORM,
        KERNEL_INITIALIZER: GLOROT_UNIFORM,
        BIAS_REGULARIZER: NONE,
        HIDDEN_LAYER: NN3_1,
        DROPOUT: False,
        DROPOUT_RATE: 0.5}, quantile=40, model_num=2, method=GEOMETRIC)
