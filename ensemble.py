# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 11. 25.
"""
import pandas as pd
from ksif.core.columns import *
from ksif import Portfolio


def cumulate(ret):
    ret.iloc[0] = 0
    ret = ret + 1
    ret = ret.cumprod()
    ret = ret - 1
    return ret


pf = Portfolio(start_date='2007-04-30')
pf.set_benchmark(KOSPI)
kospi = pf.get_benchmark().loc[:, [DATE, BENCHMARK_RET_1]]
kospi = kospi.set_index([DATE]).dropna()
kospi = cumulate(kospi)
pf.set_benchmark(KOSDAQ)
kosdaq = pf.get_benchmark().loc[:, [DATE, BENCHMARK_RET_1]]
kosdaq = kosdaq.set_index([DATE]).dropna()
kosdaq = cumulate(kosdaq)

model_num = 10
QUANTILE = 'quantile'
predicted_ret_1 = 'predict_return_1'
chunk_num = 10
labels = range(1, chunk_num + 1)

file_names = ['{}-NN3_3-all-all-linear-he_uniform-glorot_uniform-none.csv'.format(x)
              for x in range(200, 200 + model_num)]

predictions = [pd.read_csv('prediction/{}'.format(file_name)) for file_name in file_names]
selected_predictions = []
for prediction in predictions:
    prediction[QUANTILE] = prediction.groupby(by=[DATE])[predicted_ret_1].transform(
        lambda x: pd.qcut(x, chunk_num, labels=labels)
    )
    selected_predictions.append(prediction.loc[prediction[QUANTILE] == chunk_num, [DATE, CODE]])

ensemble_predictions = [selected_predictions[0]]

for current_prediction in selected_predictions[1:]:
    previous_ensemble = ensemble_predictions[-1]
    current_ensemble = pd.merge(previous_ensemble, current_prediction, on=[DATE, CODE])
    ensemble_predictions.append(current_ensemble)

for index, ensemble_prediction in enumerate(ensemble_predictions):
    ensemble_predictions[index] = pd.merge(ensemble_prediction, predictions[0].loc[:, [DATE, CODE, RET_1]], on=[DATE, CODE])

ensemble_numbers = pd.DataFrame(index=ensemble_predictions[0][DATE].unique())
ensemble_returns = pd.DataFrame(index=ensemble_predictions[0][DATE].unique())
ensemble_cumulative_returns = pd.DataFrame(index=ensemble_predictions[0][DATE].unique())
for index, ensemble_prediction in enumerate(ensemble_predictions):
    ensemble_number = ensemble_prediction.groupby(by=[DATE])[CODE].count()
    ensemble_numbers[index + 1] = ensemble_number

    ensemble_return = ensemble_prediction.groupby(by=[DATE])[RET_1].mean()
    ensemble_returns[index + 1] = ensemble_return
    ensemble_cumulative_return = cumulate(ensemble_return)
    ensemble_cumulative_returns[index + 1] = ensemble_cumulative_return
ensemble_cumulative_returns[KOSPI] = kospi
ensemble_cumulative_returns[KOSDAQ] = kosdaq

ensemble_numbers.to_csv('ensemble/ensemble_numbers.csv')
ensemble_returns.to_csv('ensemble/ensemble_returns.csv')
ensemble_cumulative_returns.to_csv('ensemble/ensemble_cumulative_returns.csv')
