# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018-09-21
"""
from model import *
from settings import *

if __name__ == '__main__':
    backtest(param={
        DATA_SET: 'value_size_momentum_quality_volatility',
        BATCH_SIZE: 300,
        EPOCHS: 100,
        ACTIVATION: TAHN,
        BIAS_INITIALIZER: ZEROS,
        KERNEL_INITIALIZER: LECUN_NORMAL,
        BIAS_REGULARIZER: NONE,
        HIDDEN_LAYER: DNN8_4,
        DROPOUT: True,
        DROPOUT_RATE: 0.5
    }, start_number=0, end_number=119, max_pool=12)
