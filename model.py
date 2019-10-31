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

from pandas.tseries.offsets import MonthEnd

TRAINING_MONTHS = 36  # After 36 months training, test 1 month.

TRAIN_START_DATE = (
        datetime.strptime(START_DATE, '%Y-%m-%d') + relativedelta(months=TRAINING_MONTHS + 1)
).strftime('%Y-%m-%d')

pf = Portfolio()
pf = pf[pf[DATE] >= START_DATE]
months = sorted(pf[DATE].unique())[:-1]

result_columns = [RET_1]

def get_rnn_predicting_set(x_test_for_rnn2, month):
    base_columns = [col for col in x_test_for_rnn2.columns if '_t-' not in col]
    factor_columns = [col for col in x_test_for_rnn2.columns if '_t-0' in col]

    rnn_columns = base_columns + sorted(factor_columns)
    x_test_for_rnn2 = x_test_for_rnn2[rnn_columns].copy(deep=True)

    rnn_predict_start_month = pd.Timestamp(month) - MonthEnd(12)
    rnn_predict_start_month = rnn_predict_start_month.strftime('%Y-%m-%d')

    x_test_for_rnn3 = x_test_for_rnn2[x_test_for_rnn2[DATE] >= rnn_predict_start_month].copy(deep=True)

    x_test_for_rnn3.sort_values([CODE, DATE], inplace=True)
    x_test_for_rnn3.reset_index(inplace=True, drop=True)

    # here, we don't have RET_1, hence it start from 2:.

    predicting_firms_list = []
    for _code, _data in x_test_for_rnn3.groupby(CODE):
        if _data.shape[0]==13:
            predicting_firms_list.append(_data.values)


    x_prediction = np.array(predicting_firms_list)
    codes = pd.DataFrame(x_prediction[:,0,1].reshape(-1,1), columns=[CODE])
    x_prediction = x_prediction[:,:,2:]

    return codes, x_prediction



def get_train_test_set(data_set_key, test_month, network_architecture):
    training_set = get_data_set(data_set_key)
    test_set = get_data_set(data_set_key)

    if network_architecture == 'rnn':
        base_columns = [col for col in training_set.columns if '_t-' not in col]
        factor_columns = [col for col in training_set.columns if '_t-0' in col]

        rnn_columns = base_columns + sorted(factor_columns)

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
            test_set = test_set.loc[test_set[DATE] == test_month, :]
        else:
            test_set = test_set.loc[(test_set[DATE] >= test_start_month) & (test_set[DATE] <= test_month), :]

    return training_set, test_set


from torch import nn
import math
import torch

class torch_DNN_noisy_net(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(torch_DNN_noisy_net, self).__init__(in_features, out_features, bias=True)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))

        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
            self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, bias)


import torch.nn.functional as F


class NoisyFactorizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=True)

        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epslion_output", torch.zeros(out_features, 1))

        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))

    def forward(self, input):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input)
        eps_out = func(self.epsilon_output)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
            noise_v = torch.mul(eps_in, eps_out)
            return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)


class NoisyDNN(nn.Module):
    def __init__(self, num_features, output_shape):
        super(NoisyDNN, self).__init__()

        self.dnn = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            torch_DNN_noisy_net(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            torch_DNN_noisy_net(32, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.dnn(x)


def train_model(month, param, early_stop, batch_normalization, minmaxscaling):
    if 'rnn' in param[HIDDEN_LAYER].lower():
        network_architecture = 'rnn'
    else:
        network_architecture = None

    data_trains, data_test = get_train_test_set(data_set_key=param[DATA_SET], test_month=month,
                                                network_architecture=network_architecture
                                                )

    if network_architecture != 'rnn':
        data_train_array = data_trains.values
        data_test_array = data_test.values

        x_train = data_train_array[:, 3:]
        y_train = data_train_array[:, 2:3]
        x_test = data_test_array[:, 3:]
        actual_test = data_test.loc[:, [DATE, CODE, RET_1]].reset_index(drop=True)
    else:
        _full_list = dict()
        _code_list = []

        data_trains.sort_values([CODE, DATE], inplace=True)
        data_test.sort_values([CODE, DATE], inplace=True)

        data_trains.reset_index(inplace=True, drop=True)
        data_test.reset_index(inplace=True, drop=True)


        for _code, _data in data_trains.groupby(CODE):
            if _data.shape[0] >= 13:
                _full_list[_code] = _data.values
                _code_list.append(_code)

        if data_test.shape[0]>0:
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

    # MinMaxScaling
    if minmaxscaling:
        year = 13
        minmaxscaling_f = lambda x: (x - x.min()) / (x.max() - x.min())
        ind = [i for i in range(len(x_train[0])) if i % year == 0]

        # x_train
        for i in ind:
            x_train[:, i:(i + year)] = minmaxscaling_f(x_train[:, i:(i + year)])
        # x_test
        for i in ind:
            x_test[:, i:(i + year)] = minmaxscaling_f(x_test[:, i:(i + year)])
        # y_train
        y_train = np.apply_along_axis(minmaxscaling_f, 0, y_train)
        # actual_test
        actual_test[RET_1] = (actual_test[RET_1] - actual_test[RET_1].min()) / (
                actual_test[RET_1].max() - actual_test[RET_1].min())

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

    if 'torch' not in param[HIDDEN_LAYER].lower():
        if network_architecture != 'rnn':
            model.add(Dense(hidden_layer[0], input_dim=input_dim,
                            activation=activation,
                            bias_initializer=bias_initializer,
                            kernel_initializer=kernel_initializer,
                            bias_regularizer=bias_regularizer
                            ))
            if batch_normalization:
                model.add(BatchNormalization())
            if dropout:
                model.add(Dropout(dropout_rate))

            for hidden_layer in hidden_layer[1:]:
                model.add(Dense(hidden_layer,
                                activation=activation,
                                bias_initializer=bias_initializer,
                                kernel_initializer=kernel_initializer
                                ))
                if batch_normalization:
                    model.add(BatchNormalization())
                if dropout:
                    model.add(Dropout(dropout_rate))

            model.add(Dense(1))
        else:
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

        if early_stop:
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=0,
                      callbacks=[EarlyStopping(patience=10)],
                      validation_split=0.2)
        else:
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=0)

    else:
        pass

    return model, x_test, actual_test


def get_file_name(param) -> str:
    file_name = '{hidden_layer}-{data_set}-{activation}-{bias_initializer}-{kernel_initializer}-{bias_regularizer}'.format(
        hidden_layer=param[HIDDEN_LAYER],
        data_set=param[DATA_SET],
        activation=param[ACTIVATION],
        bias_initializer=param[BIAS_INITIALIZER],
        kernel_initializer=param[KERNEL_INITIALIZER],
        bias_regularizer=param[BIAS_REGULARIZER],
    )
    if param[DROPOUT]:
        file_name = file_name + '-{}'.format(param[DROPOUT_RATE])

    return file_name


def get_predictions(model, x_test, actual_y=None):
    predict_ret_1 = 'predict_' + RET_1
    actual_rank = 'actual_rank'
    predicted_rank = 'predicted_rank'

    prediction = model.predict(x_test, verbose=0)

    if len(prediction.shape) == 3:
        prediction = prediction[:, -1, :]

    if isinstance(actual_y, pd.DataFrame):
        df_prediction = pd.concat(
            [actual_y,
             pd.DataFrame(prediction, columns=[predict_ret_1])],
            axis=1)
        df_prediction['diff'] = df_prediction[RET_1] - df_prediction[predict_ret_1]
        df_prediction[actual_rank] = df_prediction[RET_1].rank(ascending=False)
        df_prediction[predicted_rank] = df_prediction[predict_ret_1].rank(ascending=False)
    else:
        df_prediction = pd.DataFrame(prediction, columns=[predict_ret_1])

    return df_prediction


def backtest(param, start_number=0, end_number=9, max_pool=os.cpu_count()):
    print("Param: {}".format(param))
    pool_num = min(max_pool, end_number - start_number + 1)
    print("From {} to {} with {} processes.".format(start_number, end_number, pool_num))

    test_pf = pf.loc[pf[DATE] >= TRAIN_START_DATE, :]
    test_months = sorted(test_pf[DATE].unique())[:-1]

    with Pool(pool_num) as p:
        # for case_number in range(start_number, end_number + 1):
        #     p.apply_async(_backtest, args=(case_number, param, test_months))
        results = [p.apply_async(_backtest, (case_number, param, test_months))
                   for case_number in range(start_number, end_number + 1)]
        for r in results:
            r.wait()
        [result.get() for result in results]
        p.close()
        p.join()


def _backtest(case_number: int, param: dict, test_months: list, minmaxscaling=False,
              control_volatility_regime=False, early_stop=True, batch_normalization=True):
    tf.logging.set_verbosity(3)
    # TensorFlow wizardry
    config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    # Create a session with the above options specified.
    k.set_session(tf.Session(config=config))

    file_name = get_file_name(param)
    desc = "#{0:2d}".format(case_number)
    df_predictions = pd.DataFrame()

    # 기간설정

    for month in tqdm(test_months, desc=desc):
        if control_volatility_regime:
            # Calculate past actual volatilities
            bm = pf.get_benchmark(KOSPI)
            returns = bm.loc[:, BENCHMARK_RET_1] * 100
            returns = returns.dropna()
            window = 10
            actual_vol = returns.rolling(window).var()
            # Predict a future volatility
            ret_rolling = returns.loc[returns.index < month]
            am = arch_model(ret_rolling, vol='Garch', p=1, o=0, q=1, dist='Normal')
            res = am.fit(update_freq=0, disp='off')
            vol = res.forecast(horizon=1).variance.dropna()
            vol = vol.iloc[0, 0]
            execution = vol < actual_vol.loc[returns.index < month].quantile(.85)

            # Determine whether invest or not
            # If determined investing, train a model and get predictions.
            # Or skip this month.
            if not execution:
                continue

        if 'torch' not in param[HIDDEN_LAYER].lower():
            model, x_test, y_test = train_model(month, param, early_stop=early_stop,
                                                batch_normalization=batch_normalization, minmaxscaling=minmaxscaling)

            df_prediction = get_predictions(model, x_test, y_test)
            df_predictions = pd.concat([df_predictions, df_prediction], axis=0, ignore_index=True)
            gc.collect()

            # Clean up the memory
            k.clear_session()
        else:
            model, x_test, y_test = train_model(month, param, early_stop=early_stop,
                                                batch_normalization=batch_normalization, minmaxscaling=minmaxscaling)

    # If a directory for this model does not exist, make it.
    data_dir = 'prediction/{}'.format(file_name)
    if not Path(data_dir).exists():
        os.makedirs(data_dir)

    # Save the result of the model with a case number.
    df_predictions.to_csv(
        '{data_dir}/{case_number}-{file_name}.csv'.format(
            data_dir=data_dir,
            case_number=case_number,
            file_name=file_name
        ),
        index=False
    )

    if 'torch' not in param[HIDDEN_LAYER].lower():
        # # Clean up the memory
        k.get_session().close()
        k.clear_session()
        tf.reset_default_graph()
    else:
        pass


# noinspection PyUnresolvedReferences
def get_forward_predict(param, quantile, model_num, method):

    if 'rnn' in param[HIDDEN_LAYER].lower():
        network_architecture = 'rnn'
    else:
        network_architecture = None

    print("Param: {}".format(param))

    recent_data_set = param[DATA_SET] + '_recent'
    # x_test = pd.read_hdf('data/{}.h5'.format(recent_data_set))

    x_test = pd.read_pickle('data/{}.pck'.format(recent_data_set))
    unnamed_list = [col for col in x_test.columns if 'Unnamed:' in col]
    x_test.drop(columns=unnamed_list, inplace=True)

    x_test_for_rnn = x_test.copy(deep=True)

    month = x_test[DATE].iloc[0]
    codes = x_test[[CODE]]
    x_test = x_test.drop([DATE, CODE], axis=1)

    if network_architecture == 'rnn':
        x_test_prev = pd.read_pickle('data/{}.pck'.format(param[DATA_SET]))
        unnamed_list = [col for col in x_test_prev.columns if 'Unnamed:' in col]
        x_test_prev.drop(columns=unnamed_list, inplace=True)
        x_test_prev.drop(columns=[RET_1], inplace=True)

        x_test_for_rnn2 = pd.concat([x_test_for_rnn, x_test_prev], axis=0)

        codes, x_prediction = get_rnn_predicting_set(x_test_for_rnn2, month)


    with Pool(min(os.cpu_count(), model_num)) as p:
        # noinspection PyTypeChecker
        results = [p.apply_async(_get_forward_predict, t) for t in zip(
            [codes for _ in range(model_num)],
            [month for _ in range(model_num)],
            [param for _ in range(model_num)],
            [x_test for _ in range(model_num)],
            [x_prediction for _ in range(model_num)]
        )]
        for r in results:
            r.wait()
        results = [result.get() for result in results]
        p.close()
        p.join()

        # 0 = intersection / 1 = geometric
        get_ensemble_predictions = GET_ENSEMBLE_PREDICTIONS[method]
        ensemble_predictions = get_ensemble_predictions(results, quantile)

        ensemble_predictions = ensemble_predictions[-1][CODE]

        # Save predictions
        ensemble_predictions.to_csv('forward_predict/forward_predictions.csv', index=False)


def _get_forward_predict(codes, month, param, x_test, x_prediction, early_stop=True, batch_normalization=True, minmaxscaling=False):

    if 'rnn' in param[HIDDEN_LAYER].lower():
        network_architecture = 'rnn'
    else:
        network_architecture = None

    if 'torch' not in param[HIDDEN_LAYER].lower():
        tf.logging.set_verbosity(3)
        # TensorFlow wizardry
        config = tf.ConfigProto()
        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True
        # Create a session with the above options specified.
        k.set_session(tf.Session(config=config))

        model, _, _ = train_model(month, param, early_stop=early_stop, batch_normalization=batch_normalization,
                                  minmaxscaling=minmaxscaling)
        # get forward prediction

        if network_architecture != 'rnn':
            forward_predictions = get_predictions(model, x_test)
        else:
            forward_predictions = get_predictions(model, x_prediction)

        codes[PREDICTED_RET_1] = forward_predictions
        df_forward_predictions = codes.copy(deep=True)
        df_forward_predictions[DATE] = month

        # Clean up the memory
        k.get_session().close()
        k.clear_session()
        tf.reset_default_graph()
    else:
        print('torch would be loaded !')

    return df_forward_predictions
