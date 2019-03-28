# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2019. 3. 19.
"""
from os import listdir, makedirs
from os.path import isdir, join
from pathlib import Path

from ensemble_method import *


def get_predictions(number: int):
    prediction_path = 'prediction'
    directories = [f for f in listdir(prediction_path) if isdir(join(prediction_path, f))]

    predictions = [
        pd.read_csv(
            join(prediction_path, directory, '{}-{}.csv'.format(number, directory)), parse_dates=[DATE]
        )[[DATE, CODE, PREDICTED_RET_1]]
        for directory in directories
    ]

    return predictions


def save_adaptive_ensemble(method: str, number: int, adaptive_outcome: str,
                           long_only: bool, quantile: int, decay: float, show_plot: bool):
    assert method in METHODS
    assert number >= 0
    assert adaptive_outcome in ADAPTIVE_OUTCOMES

    predictions = get_predictions(number)

    get_ensemble_predictions = GET_ENSEMBLE_PREDICTIONS[method]

    ensemble_predictions = get_ensemble_predictions(predictions, quantile, long_only, adaptive_outcome, decay)

    # Append actual returns
    ensemble_predictions = [pd.merge(ensemble_prediction, actual_long_returns, on=[DATE, CODE])
                            for ensemble_prediction in ensemble_predictions]

    if show_plot:
        Portfolio(ensemble_predictions[-1]).outcome(weighted=WEIGHT, show_plot=True)

    folder_name = '{}-{}-{}-{}-{}'.format(method, adaptive_outcome, long_only, quantile, decay)
    file_name = '{}-{}.csv'.format(number, folder_name)
    folder_path = join('adaptive_ensemble', folder_name)

    if not Path(folder_path).exists():
        makedirs(folder_path)

    ensemble_predictions[-1].to_csv(join(folder_path, file_name), index=False)

    return ensemble_predictions


if __name__ == '__main__':
    from multiprocessing import Pool
    import os

    params = [
        # method, number, adaptive_outcome, long_only, quantile, decay, show_plot
        (BLEND, 0, SR, True, 3, 0.9, True),
        (BLEND, 0, SR, False, 3, 0.9, True),
        (ARITHMETIC, 0, SR, True, 3, 0.9, True),
        (ARITHMETIC, 0, SR, False, 3, 0.9, True),
    ]
    with Pool(min(os.cpu_count(), len(params))) as p:
        rs = [p.apply_async(save_adaptive_ensemble, param) for param in params]
        for r in rs:
            r.wait()
        p.close()
        p.join()
