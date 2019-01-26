# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018-09-23
"""
import os
from datetime import datetime
from pathlib import Path

import keras
import tensorflow as tf
from dateutil.relativedelta import relativedelta
from keras import backend as k
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential

from settings import *

TRAINING_MONTHS = 36  # After 36 months training, test 1 month.

TRAIN_START_DATE = (
        datetime.strptime(START_DATE, '%Y-%m-%d') + relativedelta(months=USED_PAST_MONTHS + TRAINING_MONTHS + 1)
).strftime('%Y-%m-%d')

pf = Portfolio()
months = sorted(pf[DATE].unique())[:-1]

result_columns = [RET_1]


def get_train_test_set(data_set_key, test_month):
    training_set = get_data_set(data_set_key)
    test_set = get_data_set(data_set_key)

    if test_month in months:
        test_index = months.index(test_month)
    else:
        test_index = len(months)
    assert test_index - USED_PAST_MONTHS - TRAINING_MONTHS >= 0, "test_month is too early"

    train_start_month = months[test_index - TRAINING_MONTHS]

    training_set = training_set.loc[(training_set[DATE] >= train_start_month) & (training_set[DATE] < test_month), :]
    test_set = test_set.loc[test_set[DATE] == test_month, :]

    return training_set, test_set


def train_model(month, param):
    data_train, data_test = get_train_test_set(data_set_key=param[DATA_SET], test_month=month)

    # Make data a numpy array
    data_train_array = data_train.values
    data_test_array = data_test.values

    X_train = data_train_array[:, 3:]
    y_train = data_train_array[:, 2:3]
    X_test = data_test_array[:, 3:]
    y_test = data_test.loc[:, [DATE, CODE, RET_1]].reset_index(drop=True)

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
              verbose=0)

    return model, X_test, y_test


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


def get_predictions(model, X, actual_y=None):
    predict_ret_1 = 'predict_' + RET_1
    actual_rank = 'actual_rank'
    predicted_rank = 'predicted_rank'

    prediction = model.predict(X, verbose=0)
    if actual_y:
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


def simulate(param, case_number):
    print("Param: {}".format(param))
    print("Case number: {}".format(case_number))

    tf.logging.set_verbosity(3)
    # TensorFlow wizardry
    config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    # Create a session with the above options specified.
    k.set_session(tf.Session(config=config))

    file_name = get_file_name(param)

    test_pf = pf.loc[pf[DATE] >= TRAIN_START_DATE, :]
    test_months = sorted(test_pf[DATE].unique())[:-1]

    df_predictions = pd.DataFrame()
    for month in tqdm(test_months):
        model, X_test, y_test = train_model(month, param)

        df_prediction = get_predictions(model, X_test, y_test)

        df_predictions = pd.concat([df_predictions, df_prediction], axis=0, ignore_index=True)

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

    # Clean up the memory
    k.get_session().close()
    k.clear_session()
    tf.reset_default_graph()


def get_forward_predict(param, quantile, model_num):
    print("Param: {}".format(param))

    tf.logging.set_verbosity(3)
    # TensorFlow wizardry
    config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    # Create a session with the above options specified.
    k.set_session(tf.Session(config=config))

    # get X_test
    RECENT_DATA_SET = param[DATA_SET] + '_recent'
    X_test = pd.read_csv('data/{}.csv'.format(RECENT_DATA_SET))
    # save month
    month = X_test[DATE][0]
    codes = X_test[[CODE]]
    X_test = X_test.drop([DATE, CODE], axis=1)

    # train model
    model, _, _ = train_model(month, param)

    # get forward prediction
    forward_predictions = get_predictions(model, X_test)
    codes['predict_return_1'] = forward_predictions
    df_forward_predictions = codes

    # ensemble

    # save forward prediction
    df_forward_predictions.to_csv('forward_predict/forward_predictions.csv')

    # Clean up the memory
    k.get_session().close()
    k.clear_session()
    tf.reset_default_graph()
