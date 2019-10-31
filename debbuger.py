# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018-09-21
"""
# noinspection PyUnresolvedReferences
from ensemble import INTERSECTION, GEOMETRIC, ARITHMETIC
from model import *
from settings import *

NET_INCOME_FILTER = 'net_income_filter'

param = {
    DATA_SET: ALL,
    BATCH_SIZE: 300,
    EPOCHS: 100,
    ACTIVATION: LINEAR,
    BIAS_INITIALIZER: HE_UNIFORM,
    KERNEL_INITIALIZER: GLOROT_UNIFORM,
    BIAS_REGULARIZER: NONE,
    HIDDEN_LAYER: RNN8_2,
    DROPOUT: False,
    DROPOUT_RATE: 0.5,
    NET_INCOME_FILTER: True
}

quantile = 40
model_num = 10
method = GEOMETRIC

# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018-09-23
"""
import gc
from datetime import datetime
from pathlib import Path
import torch

import keras
import tensorflow as tf
from arch import arch_model
from dateutil.relativedelta import relativedelta
from keras import backend as k
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential

from keras import layers

from ensemble import GET_ENSEMBLE_PREDICTIONS, PREDICTED_RET_1
from settings import *

TRAINING_MONTHS = 36  # After 36 months training, test 1 month.

TRAIN_START_DATE = (
        datetime.strptime(START_DATE, '%Y-%m-%d') + relativedelta(months=TRAINING_MONTHS + 1)
).strftime('%Y-%m-%d')

pf = Portfolio()
pf = pf[pf[DATE] >= START_DATE]
months = sorted(pf[DATE].unique())[:-1]

result_columns = [RET_1]


def get_data_set(data_set_name):
    data_set = data_sets[data_set_name]
    return data_set


data_set_key = param[DATA_SET]

training_set = get_data_set(data_set_key)
test_set = get_data_set(data_set_key)

print("Param: {}".format(param))

recent_data_set = param[DATA_SET] + '_recent'

x_test = pd.read_pickle('data/{}.pck'.format(recent_data_set))
unnamed_list = [col for col in x_test.columns if 'Unnamed:' in col]
x_test.drop(columns=unnamed_list, inplace=True)

month = x_test[DATE].iloc[0]
codes = x_test[[CODE]]
x_test = x_test.drop([DATE, CODE], axis=1)

tf.logging.set_verbosity(3)
# TensorFlow wizardry
config = tf.ConfigProto()

config.gpu_options.allow_growth = True
# Create a session with the above options specified.
k.set_session(tf.Session(config=config))

early_stop = True
batch_normalization = True
minmaxscaling = True

if 'rnn' in param[HIDDEN_LAYER].lower():
    network_architecture = 'rnn'
else:
    network_architecture = None

test_month = month

def get_train_test_set(data_set_key, test_month, network_architecture):
    training_set = get_data_set(data_set_key)
    test_set = get_data_set(data_set_key)

    if network_architecture == 'rnn':
        base_columns = [col for col in training_set.columns if '_t-' not in col]
        factor_columns = [col for col in training_set.columns if '_t-0' in col]

        rnn_columns = base_columns + factor_columns

        training_set = training_set[rnn_columns]
        test_set = test_set[rnn_columns]

    if test_month in months:
        test_index = months.index(test_month)
    else:
        test_index = len(months)
    assert test_index - TRAINING_MONTHS - 1 >= 0, "test_month is too early"

    if network_architecture != 'rnn':
        train_start_month = months[test_index - TRAINING_MONTHS - 1]
    else:
        train_start_month = months[max(test_index - TRAINING_MONTHS - 12 - 1, 0)]
    test_start_month = months[test_index - 12]

    training_set = training_set.loc[(training_set[DATE] >= train_start_month) & (training_set[DATE] < test_month), :]
    if network_architecture != 'rnn':
        test_set = test_set.loc[test_set[DATE] == test_month, :]
    else:
        if test_month not in months:
            test_set = test_set.loc[test_set[DATE] == test_month, :].copy(deep=True)
        else:
            test_set = test_set.loc[(test_set[DATE] >= test_start_month) & (test_set[DATE] <= test_month), :].copy(deep=True)

    return training_set, test_set

data_trains, data_test = get_train_test_set(data_set_key=param[DATA_SET], test_month=month,
                                            network_architecture=network_architecture
                                            )

_full_list = dict()
_code_list = []
for _code, _data in data_trains.groupby(CODE):
    if _data.shape[0] >= 13:
        _full_list[_code] = _data.values
        _code_list.append(_code)

if data_test.shape[0] > 0:
    _full_list_test = dict()
    _code_list_test = []
    for _code, _data in data_test.groupby(CODE):
        if _data.shape[0] == 13:
            _full_list_test[_code] = _data.values
            _code_list_test.append(_code)
    _code_list_final = list(set(_code_list).intersection(set(_code_list_test)))
else:
    _code_list_final = _code_list

_temp_train_list = []
for _code in _code_list_final:
    _data = _full_list[_code]
    for i in range(_data.shape[0] - 13 + 1):
        _temp_train_list.append(_data[i:i + 13])

data_train_array = np.array(_temp_train_list)

data_test_rnn = data_test.set_index(CODE)
# data_test_array_pandas = data_test_rnn.loc[_code_list_final, :].reset_index()

if data_test.shape[0] > 0:
    _temp_test_list = []
    for _code in _code_list_final:
        _data = _full_list_test[_code]
        _temp_test_list.append(_data)
    data_test_array = np.array(_temp_test_list)
else:
    data_test_array = data_test.values

x_train = data_train_array[:, :, 3:]
y_train = data_train_array[:, :, 2:3]

if data_test.shape[0] > 0:
    x_test = data_test_array[:, :, 3:]
    _actual_test = data_test_rnn[data_test_rnn[DATE] == month]

    actual_test = _actual_test.loc[_code_list_final, [DATE, RET_1]].reset_index()
else:
    x_test = None
    actual_test = None

input_length = x_train.shape[1]
input_dim = x_train.shape[-1]

# Parameters
batch_size = param[BATCH_SIZE]
epochs = param[EPOCHS]
activation = get_activation(param[ACTIVATION])
bias_initializer = get_initializer(param[BIAS_INITIALIZER])
kernel_initializer = get_initializer(param[KERNEL_INITIALIZER])
bias_regularizer = get_regularizer(param[BIAS_REGULARIZER])
hidden_layer = get_hidden_layer(param[HIDDEN_LAYER])
dropout = param[DROPOUT]
dropout_rate = param[DROPOUT_RATE]

model = Sequential()


last_layer = hidden_layer[-1]


model.add(layers.GRU(hidden_layer[0], input_shape=[input_length, input_dim],
                     activation=activation,
                     bias_initializer=bias_initializer,
                     kernel_initializer=kernel_initializer,
                     bias_regularizer=bias_regularizer,
                     return_sequences=True
                     ))
if batch_normalization:
    model.add(BatchNormalization())
if dropout:
    model.add(Dropout(dropout_rate))

for hidden_layer in hidden_layer[1:]:
    model.add(layers.GRU(hidden_layer,
                         activation=activation,
                         bias_initializer=bias_initializer,
                         kernel_initializer=kernel_initializer,
                         return_sequences=True
                         ))
    if batch_normalization:
        model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(dropout_rate))

model.add(Dense(1))

model.compile(loss=keras.losses.mse,
          optimizer=keras.optimizers.Adam())

x_test

forward_predictions = get_predictions(model, x_test_for_prediction)