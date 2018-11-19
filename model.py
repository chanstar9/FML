# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018-09-23
"""
import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from ksif import Portfolio
from ksif.core.columns import *
from scipy.stats import spearmanr
from tqdm import tqdm

from settings import *

START_DATE = '2012-05-31'

USED_PAST_MONTHS = 12  # At a time, use past 12 months data and current month data.
TRAINING_MONTHS = 36  # After 36 months training, test 1 month.

pf = Portfolio()
months = sorted(pf[DATE].unique())[:-1]

result_columns = [RET_1]


def get_train_test_set(training_set_key, test_set_key, test_month):
    training_set = get_data_set(training_set_key)
    test_set = get_data_set(test_set_key)

    test_index = months.index(test_month)
    assert test_index - USED_PAST_MONTHS - TRAINING_MONTHS >= 0, "test_month is too early"

    train_start_month = months[test_index - TRAINING_MONTHS]

    training_set = training_set.loc[(training_set[DATE] >= train_start_month) & (training_set[DATE] < test_month), :]
    test_set = test_set.loc[test_set[DATE] == test_month, :]

    return training_set, test_set


def train_model(month, param):
    tf.reset_default_graph()
    data_train, data_test = get_train_test_set(training_set_key=param[TRAINING_SET],
                                               test_set_key=param[TEST_SET], test_month=month)

    # Make data a numpy array
    data_train_array = data_train.values
    data_test_array = data_test.values

    X_train = data_train_array[:, 3:]
    y_train = data_train_array[:, 2:3]
    X_test = data_test_array[:, 3:]
    y_test = data_test_array[:, 2:3]
    actual_test = data_test.loc[:, [DATE, CODE, RET_1]].reset_index(drop=True)

    input_dim = X_train.shape[1]

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
    model.add(Dense(hidden_layer[0], input_dim=input_dim,
                    activation=activation,
                    bias_initializer=bias_initializer,
                    kernel_initializer=kernel_initializer,
                    bias_regularizer=bias_regularizer
                    ))
    model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(dropout_rate))

    for hidden_layer in hidden_layer[1:]:
        model.add(Dense(hidden_layer,
                        activation=activation,
                        bias_initializer=bias_initializer,
                        kernel_initializer=kernel_initializer
                        ))
        model.add(BatchNormalization())
        if dropout:
            model.add(Dropout(dropout_rate))

    model.add(Dense(1))
    model.compile(loss=keras.losses.mse,
                  optimizer=keras.optimizers.Adam())
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_data=(X_test, y_test))

    return model, X_test, actual_test


def get_results(model, X, actual_y):
    predict_ret_1 = 'predict_' + RET_1
    actual_rank = 'actual_rank'
    predict_rank = 'predict_rank'

    prediction = model.predict(X, verbose=0)
    df_prediction = pd.concat(
        [actual_y,
         pd.DataFrame(prediction, columns=[predict_ret_1])],
        axis=1)
    df_prediction['diff'] = df_prediction[RET_1] - df_prediction[predict_ret_1]
    df_prediction[actual_rank] = df_prediction[RET_1].rank()
    df_prediction[predict_rank] = df_prediction[predict_ret_1].rank()

    MSE = (df_prediction['diff'] ** 2).mean()
    RMSE = np.sqrt(MSE)

    CORR, _ = spearmanr(df_prediction[actual_rank], df_prediction[predict_rank])

    decile_returns = [df_prediction.loc[df_prediction[predict_rank] <= 0.1 * len(df_prediction), RET_1].mean()]
    for criteria in range(1, 10):
        decile_returns.append(df_prediction.loc[
                                  (df_prediction[predict_rank] > criteria / 10 * len(df_prediction)) &
                                  (df_prediction[predict_rank] <= (criteria + 1) * 10 * len(df_prediction)),
                                  RET_1].mean())

    return df_prediction, MSE, RMSE, CORR, decile_returns


def get_file_name(param) -> str:
    file_name = '{hidden_layer}-{training_set}-{test_set}-{activation}-{bias_initializer}-{kernel_initializer}-{bias_regularizer}'.format(
        hidden_layer=param[HIDDEN_LAYER],
        training_set=param[TRAINING_SET],
        test_set=param[TEST_SET],
        activation=param[ACTIVATION],
        bias_initializer=param[BIAS_INITIALIZER],
        kernel_initializer=param[KERNEL_INITIALIZER],
        bias_regularizer=param[BIAS_REGULARIZER],
    )

    return file_name


def simulate(param, case_number):
    file_name = get_file_name(param)

    test_pf = pf.loc[pf[DATE] >= START_DATE, :]
    test_months = sorted(test_pf[DATE].unique())[:-1]

    df_predictions = pd.DataFrame()
    MSE_list = []
    RMSE_list = []
    CORR_list = []
    decile_returns_list = []
    for month in tqdm(test_months):
        model, X_test, actual_test = train_model(month, param)

        for idx_1, layer in enumerate(model.layers):
            weights = layer.get_weights()
            for idx_2, weight in enumerate(weights):
                pd.DataFrame(weight).to_csv('weight/{}_{}_{}.csv'.format(month, idx_1, idx_2))

        df_prediction, MSE, RMSE, CORR, decile_returns = get_results(
            model, X_test, actual_test
        )

        df_predictions = pd.concat([df_predictions, df_prediction], axis=0, ignore_index=True)
        MSE_list.append(MSE)
        RMSE_list.append(RMSE)
        CORR_list.append(CORR)
        decile_returns_list.append(decile_returns)

    decile_returns_list = np.array(decile_returns_list)
    result_data = {
        DATE: test_months,
        'MSE': MSE_list,
        'RMSE': RMSE_list,
        'CORR': CORR_list
    }
    for index in range(0, 10):
        result_data[index + 1] = decile_returns_list[:, index]
    df_result = pd.DataFrame(data=result_data)

    df_predictions.to_csv(
        'prediction/{case_number}-{file_name}.csv'.format(case_number=case_number, file_name=file_name), index=False)
    df_result.to_csv('result/{case_number}-{file_name}.csv'.format(case_number=case_number, file_name=file_name),
                     index=False)
