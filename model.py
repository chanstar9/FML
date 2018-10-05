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

pf = Portfolio()
pf = pf.loc[pf[DATE] <= '2018-07-31', :]
months = sorted(pf[DATE].unique())

result_columns = [RET_1]


def get_train_test_set(data_set, test_month):
    test_index = months.index(test_month)
    assert test_index - 12 - 36 >= 0, "test_month is too early"

    train_start_month = months[test_index - 36]

    training_set = data_set.loc[(data_set[DATE] >= train_start_month) & (data_set[DATE] < test_month), :]
    test_set = data_set.loc[data_set[DATE] == test_month, :]

    return training_set, test_set


def train_model(month, param):
    tf.reset_default_graph()
    data_train, data_test = get_train_test_set(data_set=get_data_set(param[DATA_SET]), test_month=month)

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

    top_tertile_return = df_prediction.loc[df_prediction[predict_rank] > 0.6666 * len(df_prediction), RET_1].mean()
    assert pd.notna(top_tertile_return)
    bottom_tertile_return = df_prediction.loc[df_prediction[predict_rank] < 0.3333 * len(df_prediction), RET_1].mean()
    assert pd.notna(bottom_tertile_return)
    long_short_tertile_return = top_tertile_return - bottom_tertile_return
    assert pd.notna(long_short_tertile_return)

    top_quintile_return = df_prediction.loc[df_prediction[predict_rank] > 0.8 * len(df_prediction), RET_1].mean()
    assert pd.notna(top_quintile_return)
    bottom_quintile_return = df_prediction.loc[df_prediction[predict_rank] < 0.2 * len(df_prediction), RET_1].mean()
    assert pd.notna(bottom_quintile_return)
    long_short_quintile_return = top_quintile_return - bottom_quintile_return
    assert pd.notna(long_short_quintile_return)

    return df_prediction, MSE, RMSE, CORR, top_tertile_return, long_short_tertile_return, bottom_tertile_return, \
           top_quintile_return, long_short_quintile_return, bottom_quintile_return


def get_file_name(param) -> str:
    file_name = '{hidden_layer}({data_set}_{activation}_{bias_initializer}_{kernel_initializer}_{bias_regularizer})'.format(
        hidden_layer=param[HIDDEN_LAYER],
        data_set=param[DATA_SET],
        activation=param[ACTIVATION],
        bias_initializer=param[BIAS_INITIALIZER],
        kernel_initializer=param[KERNEL_INITIALIZER],
        bias_regularizer=param[BIAS_REGULARIZER],
    )

    return file_name


def simulate(param, case_number):
    file_name = get_file_name(param)

    test_pf = pf.loc[pf[DATE] >= '2012-05-31', :]
    test_months = sorted(test_pf[DATE].unique())

    df_predictions = pd.DataFrame()
    MSE_list = []
    RMSE_list = []
    CORR_list = []
    long_short_tertile_return_list = []
    top_tertile_return_list = []
    bottom_tertile_return_list = []
    long_short_quintile_return_list = []
    top_quintile_return_list = []
    bottom_quintile_return_list = []
    for month in tqdm(test_months):
        model, X_test, actual_test = train_model(month, param)

        df_prediction, MSE, RMSE, CORR, top_tertile_return, long_short_tertile_return, bottom_tertile_return, \
        top_quintile_return, long_short_quintile_return, bottom_quintile_return = get_results(
            model, X_test, actual_test
        )

        df_predictions = pd.concat([df_predictions, df_prediction], axis=0, ignore_index=True)
        MSE_list.append(MSE)
        RMSE_list.append(RMSE)
        CORR_list.append(CORR)
        long_short_tertile_return_list.append(long_short_tertile_return)
        top_tertile_return_list.append(top_tertile_return)
        bottom_tertile_return_list.append(bottom_tertile_return)
        long_short_quintile_return_list.append(long_short_quintile_return)
        top_quintile_return_list.append(top_quintile_return)
        bottom_quintile_return_list.append(bottom_quintile_return)

    df_result = pd.DataFrame(data={
        DATE: test_months,
        'MSE': MSE_list,
        'RMSE': RMSE_list,
        'CORR': CORR_list,
        'top_tertile_return': top_tertile_return_list,
        'long_short_tertile_return': long_short_tertile_return_list,
        'bottom_tertile_return': bottom_tertile_return_list,
        'top_quintile_return': top_quintile_return_list,
        'long_short_quintile_return': long_short_quintile_return_list,
        'bottom_quintile_return': bottom_quintile_return_list,
    })

    df_predictions.to_csv(
        'prediction/{case_number}_{file_name}.csv'.format(case_number=case_number, file_name=file_name), index=False)
    df_result.to_csv('result/{case_number}_{file_name}.csv'.format(case_number=case_number, file_name=file_name),
                     index=False)
