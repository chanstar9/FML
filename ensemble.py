# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 11. 25.
"""
import pandas as pd
import numpy as np
from ksif.core.columns import *
from ksif import Portfolio
import matplotlib.pyplot as plt

INTERSECTION = 'Intersection'
GEOMETRIC = 'Geometric'

QUANTILE = 'quantile'
PREDICTED_RET_1 = 'predict_return_1'


def get_intersection_ensemble_predictions(model_name: str, start_number: int = 0, end_number: int = 9,
                                          quantile: int = 40):
    """
    :return ensemble_predictions:
        DATE    | (datetime64)
        CODE    | (str)
        RET_1   | (float64)
    """

    predictions = get_predictions(model_name, start_number, end_number)
    selected_predictions = select_predictions(predictions, quantile, [DATE, CODE])

    # Intersection
    ensemble_predictions = [selected_predictions[0]]
    for current_prediction in selected_predictions[1:]:
        previous_ensemble = ensemble_predictions[-1]
        current_ensemble = pd.merge(previous_ensemble, current_prediction, on=[DATE, CODE])
        ensemble_predictions.append(current_ensemble)
    for index, ensemble_prediction in enumerate(ensemble_predictions):
        ensemble_predictions[index] = pd.merge(
            ensemble_prediction, predictions[0].loc[:, [DATE, CODE, RET_1]], on=[DATE, CODE]
        )
    return ensemble_predictions


def get_geometric_ensemble_predictions(model_name: str, start_number: int = 0, end_number: int = 9,
                                       quantile: int = 40):
    """
    :return ensemble_predictions:
        DATE    | (datetime64)
        CODE    | (str)
        RET_1   | (float64)
    """
    predictions = get_predictions(model_name, start_number, end_number)

    # Take exponential
    for prediction in predictions:
        prediction[PREDICTED_RET_1] = np.exp(prediction[PREDICTED_RET_1])

    # Geometric mean
    ensemble_predictions = [predictions[0]]
    for current_prediction in predictions[1:]:
        previous_ensemble = ensemble_predictions[-1]
        current_ensemble = current_prediction
        current_ensemble[PREDICTED_RET_1] = previous_ensemble[PREDICTED_RET_1] * current_prediction[PREDICTED_RET_1]
        ensemble_predictions.append(current_ensemble)
    for index, ensemble_prediction in enumerate(ensemble_predictions):
        ensemble_prediction[PREDICTED_RET_1] = ensemble_prediction[PREDICTED_RET_1] ** (1 / (index + 1))

    # Take log
    for ensemble_prediction in ensemble_predictions:
        ensemble_prediction[PREDICTED_RET_1] = np.log(ensemble_prediction[PREDICTED_RET_1])

    # Select the top quantile
    ensemble_predictions = select_predictions(ensemble_predictions, quantile, [DATE, CODE, RET_1])

    return ensemble_predictions


def select_predictions(predictions, quantile, columns):
    labels = range(1, quantile + 1)
    selected_predictions = []
    for prediction in predictions:
        prediction[QUANTILE] = prediction.groupby(by=[DATE])[PREDICTED_RET_1].transform(
            lambda x: pd.qcut(x, quantile, labels=labels)
        )
        selected_predictions.append(prediction.loc[prediction[QUANTILE] == quantile, columns])
    return selected_predictions


def get_predictions(model_name, start_number, end_number):
    file_names = [
        '{}-{}.csv'.format(x, model_name) for x in range(start_number, end_number + 1)
    ]
    predictions = [
        pd.read_csv('prediction/{}/{}'.format(model_name, file_name), parse_dates=[DATE]) for file_name in file_names
    ]
    return predictions


METHODS = [
    INTERSECTION,
    GEOMETRIC
]

GET_ENSEMBLE_PREDICTIONS = {
    INTERSECTION: get_intersection_ensemble_predictions,
    GEOMETRIC: get_geometric_ensemble_predictions
}


def _cumulate(ret):
    ret.iloc[0] = 0
    ret = ret + 1
    ret = ret.cumprod()
    ret = ret - 1
    return ret


def plot_intersection_ensemble(method: str, model_name: str, start_number: int = 0, end_number: int = 9, step: int = 1,
                               quantile: int = 40) -> None:
    # Check parameters
    assert method in METHODS, "method does not exist."
    assert end_number > start_number + 1, "end_number should be bigger than (start_number + 1)."
    assert step >= 1, "step should be a positive integer."
    assert quantile > 1, "quantile should be an integer bigger than 1."

    result_file_name = 'summary/{}_ensemble/{}-{}'.format(method.lower(), quantile, model_name)

    get_ensemble_predictions = GET_ENSEMBLE_PREDICTIONS[method]

    ensemble_predictions = get_ensemble_predictions(model_name, start_number, end_number, quantile)

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
    axes[0].set_title('{}:{}, Top {}-quantile'.format(method, model_name, quantile), fontdict={'fontsize': 16})
    axes[0].set_ylabel('# of companies')
    axes[0].legend(loc='upper left')
    ensemble_cumulative_returns.plot(ax=axes[1], colormap='Blues')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Return')
    axes[1].legend(loc='upper left')
    plt.savefig(result_file_name + '.png')
    fig.show('summary/{}_ensemble/{}.csv'.format(method.lower(), model_name))

    # Summary
    for ensemble_prediction in ensemble_predictions:
        ensemble_prediction[DATE] = ensemble_prediction[DATE].astype(str)
    ensemble_portfolios = [Portfolio(ensemble_prediction) for ensemble_prediction in
                           ensemble_predictions[(step - 1)::step]]
    ensemble_outcomes = [ensemble_portfolio.outcome() for ensemble_portfolio in
                         ensemble_portfolios[(step - 1)::step]]
    total_returns = [ensemble_outcome['total_return'] for ensemble_outcome in
                     ensemble_outcomes[(step - 1)::step]]
    active_returns = [ensemble_outcome['active_return'] for ensemble_outcome in
                      ensemble_outcomes[(step - 1)::step]]
    active_risks = [ensemble_outcome['active_risk'] for ensemble_outcome in
                    ensemble_outcomes[(step - 1)::step]]
    information_ratios = [ensemble_outcome['information_ratio'] for ensemble_outcome in
                          ensemble_outcomes[(step - 1)::step]]
    ensemble_summary = pd.DataFrame({
        'total_return': total_returns,
        'active_return': active_returns,
        'active_risk': active_risks,
        'information_ratio': information_ratios
    }, index=ensemble_numbers.columns)
    ensemble_summary.to_csv(result_file_name + '.csv')


if __name__ == '__main__':
    models = [
        'NN3_3-all-linear-he_uniform-glorot_uniform-none',
        'DNN8_1-all-linear-he_uniform-glorot_uniform-none',
        'DNN8_1-all-linear-he_uniform-glorot_uniform-none-0.5',
        'DNN8_2-all-linear-he_uniform-glorot_uniform-none'
    ]

    for method in METHODS:
        for model in models:
            plot_intersection_ensemble(method, model)
