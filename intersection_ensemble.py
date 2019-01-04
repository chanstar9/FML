# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 11. 25.
"""
import pandas as pd
from ksif.core.columns import *
import matplotlib.pyplot as plt

QUANTILE = 'quantile'
PREDICTED_RET_1 = 'predict_return_1'


def _cumulate(ret):
    ret.iloc[0] = 0
    ret = ret + 1
    ret = ret.cumprod()
    ret = ret - 1
    return ret


def plot_intersection_ensemble(model_name, start_number, end_number, step=1, quantile=20):
    labels = range(1, quantile + 1)
    file_names = ['{}-{}.csv'.format(x, model_name)
                  for x in range(start_number, end_number + 1)]
    predictions = [pd.read_csv('prediction/{}/{}'.format(model_name, file_name), parse_dates=[DATE]) for file_name in file_names]
    selected_predictions = []
    for prediction in predictions:
        prediction[QUANTILE] = prediction.groupby(by=[DATE])[PREDICTED_RET_1].transform(
            lambda x: pd.qcut(x, quantile, labels=labels)
        )
        selected_predictions.append(prediction.loc[prediction[QUANTILE] == quantile, [DATE, CODE]])
    ensemble_predictions = [selected_predictions[0]]

    for current_prediction in selected_predictions[1:]:
        previous_ensemble = ensemble_predictions[-1]
        current_ensemble = pd.merge(previous_ensemble, current_prediction, on=[DATE, CODE])
        ensemble_predictions.append(current_ensemble)

    for index, ensemble_prediction in enumerate(ensemble_predictions):
        ensemble_predictions[index] = pd.merge(ensemble_prediction, predictions[0].loc[:, [DATE, CODE, RET_1]],
                                               on=[DATE, CODE])

    # Cumulative ensemble
    ensemble_numbers = pd.DataFrame(index=ensemble_predictions[0][DATE].unique())
    ensemble_cumulative_returns = pd.DataFrame(index=ensemble_predictions[0][DATE].unique())
    for index, ensemble_prediction in enumerate(ensemble_predictions):
        ensemble_number = ensemble_prediction.groupby(by=[DATE])[CODE].count()
        ensemble_return = ensemble_prediction.groupby(by=[DATE])[RET_1].mean()
        ensemble_cumulative_return = _cumulate(ensemble_return)

        if (index + 1) % step == 0:
            ensemble_numbers[index + 1] = ensemble_number
            ensemble_cumulative_returns[index + 1] = ensemble_cumulative_return

    # Fill nan
    ensemble_numbers.fillna(0, inplace=True)
    ensemble_cumulative_returns.fillna(method='ffill', inplace=True)
    ensemble_cumulative_returns.fillna(0, inplace=True)

    # Plot
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
    ensemble_numbers.plot(ax=axes[0], colormap='Blues')
    axes[0].set_title('Model:{}, Top {}-quantile'.format(model_name, quantile), fontdict={'fontsize': 16})
    axes[0].set_ylabel('# of companies')
    axes[0].legend(loc='upper left')
    ensemble_cumulative_returns.plot(ax=axes[1], colormap='Blues')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Return')
    axes[1].legend(loc='upper left')
    fig.show()


if __name__ == '__main__':
    start_number = 0
    end_number = 9

    model_name = 'NN3_3-all-all-linear-he_uniform-glorot_uniform-none'
    plot_intersection_ensemble(model_name, start_number, end_number, step=1)

    model_name = 'DNN8_4-all-all-relu-he_uniform-glorot_uniform-none'
    plot_intersection_ensemble(model_name, start_number, end_number, step=1)
