import pandas as pd
from ksif import *

from settings import ADAPTIVE_WINDOW

RANK = 'rank'
PREDICTED_RET_1 = 'predict_return_1'
WEIGHT = 'weight'

BLEND = 'blend'
ARITHMETIC = 'arithmetic'

NONE = 'none'

pf = Portfolio()
CD91_returns = pf.get_benchmark(CD91)[BENCHMARK_RET_1]
CD91_returns = CD91_returns.dropna()

actual_long_returns = pf[[DATE, CODE, RET_1]]


def _select_predictions(predictions, quantile, is_long):
    """
    :return selected_predictions:
        DATE        | (datetime64)
        CODE        | (str)
        RET_1       | (float)
        WEIGHT      | (float)
    """
    selected_predictions = []
    for prediction in predictions:
        prediction.loc[:, RANK] = prediction.groupby(by=[DATE])[PREDICTED_RET_1].transform(
            lambda x: x.rank(ascending=not is_long, pct=True)
        )
        # Append actual returns
        prediction = pd.merge(prediction, actual_long_returns, on=[DATE, CODE])
        prediction.loc[:, WEIGHT] = 1
        selected_predictions.append(prediction.loc[prediction[RANK] <= (1 / quantile), [DATE, CODE, RET_1, WEIGHT]])
    return selected_predictions


def _concat_ensemble_predictions(ensemble_long_predictions, ensemble_short_predictions):
    ensemble_predictions = [
        pd.concat(
            [ensemble_long_prediction, ensemble_short_prediction], axis=0
        ).groupby([DATE, CODE]).sum().reset_index(drop=False)
        for ensemble_long_prediction, ensemble_short_prediction
        in zip(ensemble_long_predictions, ensemble_short_predictions)
    ]
    return ensemble_predictions


def get_blend_ensemble_predictions(predictions, quantile, long_only, adaptive_outcome, decay):
    """
    :return ensemble_predictions:
        DATE        | (datetime64)
        CODE        | (str)
        WEIGHT      | (float)
    """
    selected_long_predictions = _select_predictions(predictions, quantile, is_long=True)
    if long_only:

        if adaptive_outcome is not NONE:
            selected_portfolios = [Portfolio(selected_prediction) for selected_prediction in selected_long_predictions]
            cumulative_outcomes = _get_cumulative_outcomes(decay, adaptive_outcome, selected_portfolios)
            selected_long_predictions = _apply_adaptive_weights(selected_long_predictions, cumulative_outcomes,
                                                                adaptive_outcome)

        ensemble_predictions = _get_blend_ensemble_predictions(selected_long_predictions, is_long=True)

    else:
        selected_short_predictions = _select_predictions(predictions, quantile, is_long=False)

        if adaptive_outcome is not NONE:
            selected_predictions = _concat_ensemble_predictions(selected_long_predictions, selected_short_predictions)
            selected_portfolios = [Portfolio(selected_prediction) for selected_prediction in selected_predictions]
            cumulative_outcomes = _get_cumulative_outcomes(decay, adaptive_outcome, selected_portfolios)
            selected_long_predictions = _apply_adaptive_weights(selected_long_predictions, cumulative_outcomes,
                                                                adaptive_outcome)
            selected_short_predictions = _apply_adaptive_weights(selected_short_predictions, cumulative_outcomes,
                                                                 adaptive_outcome)

        ensemble_long_predictions = _get_blend_ensemble_predictions(selected_long_predictions, is_long=True)
        ensemble_short_predictions = _get_blend_ensemble_predictions(selected_short_predictions, is_long=False)
        ensemble_predictions = _concat_ensemble_predictions(ensemble_long_predictions, ensemble_short_predictions)

    ensemble_predictions = [ensemble_prediction[[DATE, CODE, WEIGHT]] for ensemble_prediction in ensemble_predictions]
    return ensemble_predictions


def _apply_adaptive_weights(selected_predictions, cumulative_outcomes, adaptive_outcome):
    selected_predictions = [
        selected_prediction.merge(cumulative_outcome, how='inner', on=DATE)
        for selected_prediction, cumulative_outcome in zip(selected_predictions, cumulative_outcomes)
    ]
    for selected_prediction in selected_predictions:
        selected_prediction.loc[:, WEIGHT] = \
            selected_prediction.loc[:, WEIGHT] * selected_prediction.loc[:, adaptive_outcome]
    return selected_predictions


def _get_cumulative_outcomes(decay, outcome, selected_portfolios):
    cumulative_outcomes = [
        pd.DataFrame(
            data={
                outcome: [
                    max(selected_portfolio.loc[
                        (start_date <= selected_portfolio[DATE]) & (selected_portfolio[DATE] < end_date), :
                        ].outcome(weighted=WEIGHT)[outcome], 0)
                    for start_date, end_date in zip(
                        sorted(selected_portfolio[DATE].unique())[:-ADAPTIVE_WINDOW],
                        sorted(selected_portfolio[DATE].unique())[ADAPTIVE_WINDOW:]
                    )
                ]}, index=pd.Series(sorted(selected_portfolio[DATE].unique())[ADAPTIVE_WINDOW:], name=DATE))
        for selected_portfolio in selected_portfolios
    ]
    decays = pd.Series(data=[decay ** index for index in range(len(cumulative_outcomes[0]), 0, -1)],
                       index=cumulative_outcomes[0].index, name='decay')
    for cumulative_outcome in cumulative_outcomes:
        cumulative_outcome[outcome] = (cumulative_outcome[outcome] * decays).cumsum() / decays

    return cumulative_outcomes


def _get_blend_ensemble_predictions(selected_predictions, is_long):
    # Add weights
    sign = 1.0 if is_long else -1.0
    for selected_prediction in selected_predictions:
        selected_prediction.loc[:, WEIGHT] = selected_prediction.loc[:, WEIGHT] * sign
    # Intersection
    ensemble_predictions = [selected_predictions[0][[DATE, CODE, WEIGHT]]]
    for current_prediction in selected_predictions[1:]:
        previous_ensemble = ensemble_predictions[-1].loc[:, [DATE, CODE, WEIGHT]].set_index([DATE, CODE])
        current_ensemble = current_prediction.loc[:, [DATE, CODE, WEIGHT]].set_index([DATE, CODE]).add(
            previous_ensemble, fill_value=0)
        current_ensemble.reset_index(drop=False, inplace=True)
        ensemble_predictions.append(current_ensemble)
    return ensemble_predictions


def get_arithmetic_ensemble_predictions(predictions, quantile, long_only, adaptive_outcome, decay):
    """
    :return ensemble_predictions: ([DataFrame])
        DATE        | (datetime64)
        CODE        | (str)
        WEIGHT      | (float)
    """
    # Calculate cumulative outcomes
    if adaptive_outcome is NONE:
        cumulative_outcomes = None
    else:
        selected_long_predictions = _select_predictions(predictions, quantile, is_long=True)
        if long_only:
            selected_portfolios = [Portfolio(selected_prediction) for selected_prediction in selected_long_predictions]
            cumulative_outcomes = _get_cumulative_outcomes(decay, adaptive_outcome, selected_portfolios)

        else:
            selected_short_predictions = _select_predictions(predictions, quantile, is_long=False)
            selected_predictions = _concat_ensemble_predictions(selected_long_predictions, selected_short_predictions)
            selected_portfolios = [Portfolio(selected_prediction) for selected_prediction in selected_predictions]
            cumulative_outcomes = _get_cumulative_outcomes(decay, adaptive_outcome, selected_portfolios)

    ensemble_long_predictions = _get_arithmetic_ensemble_predictions(predictions, quantile, True,
                                                                     cumulative_outcomes, adaptive_outcome)
    if long_only:
        ensemble_predictions = ensemble_long_predictions
    else:
        ensemble_short_predictions = _get_arithmetic_ensemble_predictions(predictions, quantile, False,
                                                                          cumulative_outcomes, adaptive_outcome)
        ensemble_predictions = _concat_ensemble_predictions(ensemble_long_predictions, ensemble_short_predictions)

    ensemble_predictions = [ensemble_prediction[[DATE, CODE, WEIGHT]] for ensemble_prediction in ensemble_predictions]
    return ensemble_predictions


CUMULATIVE_WEIGHT_SUM = 'cumulative_weight_sum'


def _get_arithmetic_ensemble_predictions(predictions, quantile, is_long, cumulative_outcomes, adaptive_outcome):
    # Add weight
    for prediction in predictions:
        prediction.loc[:, WEIGHT] = 1

    if adaptive_outcome is not NONE:
        predictions = _apply_adaptive_weights(predictions, cumulative_outcomes, adaptive_outcome)

    # Arithmetic mean
    predictions[0][PREDICTED_RET_1] = predictions[0][PREDICTED_RET_1] * predictions[0][WEIGHT]
    predictions[0][CUMULATIVE_WEIGHT_SUM] = predictions[0][WEIGHT]
    ensemble_predictions = [predictions[0]]
    for current_prediction in predictions[1:]:
        previous_ensemble = ensemble_predictions[-1]
        current_ensemble = current_prediction
        current_ensemble[PREDICTED_RET_1] = \
            previous_ensemble[PREDICTED_RET_1] * previous_ensemble[WEIGHT] + \
            current_prediction[PREDICTED_RET_1] * current_prediction[WEIGHT]
        current_ensemble[CUMULATIVE_WEIGHT_SUM] = previous_ensemble[CUMULATIVE_WEIGHT_SUM] + current_ensemble[WEIGHT]
        ensemble_predictions.append(current_ensemble)
    for ensemble_prediction in ensemble_predictions:
        ensemble_prediction[PREDICTED_RET_1] = \
            ensemble_prediction[PREDICTED_RET_1] / ensemble_prediction[CUMULATIVE_WEIGHT_SUM]
    # Select the top quantile
    ensemble_predictions = _select_predictions(ensemble_predictions, quantile, is_long=is_long)

    # Apply sign
    sign = 1.0 if is_long else -1.0
    for ensemble_prediction in ensemble_predictions:
        ensemble_prediction.loc[:, WEIGHT] = ensemble_prediction.loc[:, WEIGHT] * sign

    return ensemble_predictions


METHODS = [
    BLEND,
    ARITHMETIC
]

ADAPTIVE_OUTCOMES = [
    NONE,
    CAGR,
    SR,
    IR
]

GET_ENSEMBLE_PREDICTIONS = {
    BLEND: get_blend_ensemble_predictions,
    ARITHMETIC: get_arithmetic_ensemble_predictions
}
