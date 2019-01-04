# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018-09-21
"""
from model import *
from settings import *

# params = []
#
# for data_set in sorted(data_sets.keys()):
#     for activation in sorted(activations.keys()):
#         for bias_initializer in sorted(initializers.keys()):
#             for kernel_initializer in sorted(initializers.keys()):
#                 for bias_regularizer in sorted(regularizers.keys()):
#                     for hidden_layer in sorted(hidden_layers.keys()):
#                         params.append({
#                             TRAINING_SET: data_set,
#                             TEST_SET: data_set,
#                             BATCH_SIZE: 300,
#                             EPOCHS: 100,
#                             ACTIVATION: activation,
#                             BIAS_INITIALIZER: bias_initializer,
#                             KERNEL_INITIALIZER: kernel_initializer,
#                             BIAS_REGULARIZER: bias_regularizer,
#                             HIDDEN_LAYER: hidden_layer,
#                             DROPOUT: False,
#                             DROPOUT_RATE: 0.5
#                         })

if __name__ == '__main__':

    # case_number = 0
    # print('Case: {}'.format(case_number))
    # print('Param: {}'.format(params[case_number]))
    # simulate(param=params[case_number], case_number=case_number)

    simulate(param={
        DATA_SET: ALL,
        BATCH_SIZE: 300,
        EPOCHS: 100,
        ACTIVATION: LINEAR,
        BIAS_INITIALIZER: HE_UNIFORM,
        KERNEL_INITIALIZER: GLOROT_UNIFORM,
        BIAS_REGULARIZER: NONE,
        HIDDEN_LAYER: NN3_3,
        DROPOUT: False,
        DROPOUT_RATE: 0.5
    }, case_number=0)
