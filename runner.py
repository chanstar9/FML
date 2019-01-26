# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018-09-21
"""
import gc

from model import *
from settings import *

if __name__ == '__main__':
    for i in range(0, 10):
        simulate(param={
            DATA_SET: ALL,
            BATCH_SIZE: 300,
            EPOCHS: 100,
            ACTIVATION: LINEAR,
            BIAS_INITIALIZER: HE_UNIFORM,
            KERNEL_INITIALIZER: GLOROT_UNIFORM,
            BIAS_REGULARIZER: NONE,
            HIDDEN_LAYER: NN3_1,
            DROPOUT: False,
            DROPOUT_RATE: 0.5
        }, case_number=i)
        gc.collect()

    get_forward_predict(param={
        DATA_SET: ALL,
        BATCH_SIZE: 300,
        EPOCHS: 100,
        ACTIVATION: LINEAR,
        BIAS_INITIALIZER: HE_UNIFORM,
        KERNEL_INITIALIZER: GLOROT_UNIFORM,
        BIAS_REGULARIZER: NONE,
        HIDDEN_LAYER: NN3_1,
        DROPOUT: False,
        DROPOUT_RATE: 0.5}, quantile=40, model_num=10)
