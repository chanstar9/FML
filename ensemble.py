# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 11. 25.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm

from ensemble_method import *

QUANTILE = 'quantile'
COUNT = 'count'

CORRECT = 'correct'


def get_predictions(model_name, start_number, end_number):
    file_names = [
        '{}-{}.csv'.format(x, model_name) for x in range(start_number, end_number + 1)
    ]
    predictions = [
        pd.read_csv('prediction/{}/{}'.format(model_name, file_name), parse_dates=[DATE])[[DATE, CODE, PREDICTED_RET_1]]
        for file_name in file_names
    ]
    return predictions


def _get_file_name(method: str, model_name: str, quantile: int) -> str:
    result_file_name = '{}/{}-{}'.format(method.lower(), quantile, model_name)
    return result_file_name


def get_ensemble(method: str, adaptive_outcome: str, model_name: str, start_number: int, end_number: int, step: int,
                 long_only: bool, quantile: int, decay: float, show_plot):
    """

    :param method: (str)
    :param adaptive_outcome: (str)
    :param model_name: (str)
    :param start_number: (int)
    :param end_number: (int)
    :param step: (int)
    :param long_only: (bool)
    :param quantile: (int)
    :param decay: (float)
    :param show_plot: (bool)

    :return ensemble_summary: (DataFrame)
        PORTFOLIO_RETURN    | (float)
        ACTIVE_RETURN       | (float)
        ACTIVE_RISK         | (float)
        IR                  | (float)
        CAGR                | (float)
    :return ensemble_portfolios: ([Portfolio])
        DATE                | (datetime)
        CODE                | (str)
        RET_1               | (float)
    """
    # Check parameters
    assert method in METHODS, "method does not exist."
    assert adaptive_outcome in ADAPTIVE_OUTCOMES, "outcome does not exist."
    assert end_number > start_number, "end_number should be bigger than (start_number + 1)."
    assert step >= 1, "step should be a positive integer."
    assert quantile > 1, "quantile should be an integer bigger than 1."

    result_file_name = _get_file_name(method, model_name, quantile)

    predictions = get_predictions(model_name, start_number, end_number)

    get_ensemble_predictions = GET_ENSEMBLE_PREDICTIONS[method]

    ensemble_predictions = get_ensemble_predictions(predictions, quantile, long_only, adaptive_outcome, decay)
    if long_only:
        ensemble_predictions = [ensemble_prediction.loc[ensemble_prediction[WEIGHT] > 0, :] for
                                ensemble_prediction in ensemble_predictions]

    # Append actual returns
    ensemble_predictions = [pd.merge(ensemble_prediction, actual_long_returns, on=[DATE, CODE]) for
                            ensemble_prediction in ensemble_predictions]

    ensemble_portfolios = [Portfolio(ensemble_prediction) for ensemble_prediction in
                           ensemble_predictions[(step - 1)::step]]

    for ensemble_portfolio in ensemble_portfolios:
        if ensemble_portfolio.empty:
            return None, None

    ensemble_outcomes = [ensemble_portfolio.outcome(weighted=WEIGHT)
                         for ensemble_portfolio in ensemble_portfolios]
    portfolio_returns = [ensemble_outcome[PORTFOLIO_RETURN] for ensemble_outcome in ensemble_outcomes]
    active_returns = [ensemble_outcome[ACTIVE_RETURN] for ensemble_outcome in ensemble_outcomes]
    active_risks = [ensemble_outcome[ACTIVE_RISK] for ensemble_outcome in ensemble_outcomes]
    information_ratios = [ensemble_outcome[IR] for ensemble_outcome in ensemble_outcomes]
    sharpe_ratios = [ensemble_outcome[SR] for ensemble_outcome in ensemble_outcomes]
    MDDs = [ensemble_outcome[MDD] for ensemble_outcome in ensemble_outcomes]
    alphas = [ensemble_outcome[FAMA_FRENCH_ALPHA] for ensemble_outcome in ensemble_outcomes]
    alpha_p_values = [ensemble_outcome[FAMA_FRENCH_ALPHA_P_VALUE] for ensemble_outcome in ensemble_outcomes]
    betas = [ensemble_outcome[FAMA_FRENCH_BETA] for ensemble_outcome in ensemble_outcomes]
    CAGRs = [ensemble_outcome[CAGR] for ensemble_outcome in ensemble_outcomes]

    ensemble_summary = pd.DataFrame({
        PORTFOLIO_RETURN: portfolio_returns,
        ACTIVE_RETURN: active_returns,
        ACTIVE_RISK: active_risks,
        IR: information_ratios,
        SR: sharpe_ratios,
        MDD: MDDs,
        FAMA_FRENCH_ALPHA: alphas,
        FAMA_FRENCH_ALPHA_P_VALUE: alpha_p_values,
        FAMA_FRENCH_BETA: betas,
        CAGR: CAGRs,
    }, index=range(step - 1, end_number - start_number + 1, step))

    if not Path('summary/' + method).exists():
        os.makedirs('summary/' + method)

    ensemble_summary.to_csv('summary/' + result_file_name + '.csv')
    for ensemble_prediction in ensemble_predictions:
        ensemble_prediction[DATE] = pd.to_datetime(ensemble_prediction[DATE], format='%Y-%m-%d')

    # Plot
    if show_plot:
        # Cumulative ensemble
        ensemble_long_numbers = pd.DataFrame(index=ensemble_predictions[0][DATE].unique())
        ensemble_short_numbers = pd.DataFrame(index=ensemble_predictions[0][DATE].unique())
        ensemble_cumulative_returns = pd.DataFrame(index=ensemble_predictions[0][DATE].unique())
        for index, ensemble_portfolio in enumerate(ensemble_portfolios):
            long_portfolio = ensemble_portfolio.loc[ensemble_portfolio[WEIGHT] > 0, :]
            short_portfolio = ensemble_portfolio.loc[ensemble_portfolio[WEIGHT] < 0, :]
            short_portfolio.loc[:, RET_1] = -1 * short_portfolio.loc[:, RET_1]
            short_portfolio.loc[:, WEIGHT] = -short_portfolio.loc[:, WEIGHT]
            ensemble_long_number = long_portfolio.groupby(by=[DATE])[CODE].count()
            ensemble_short_number = short_portfolio.groupby(by=[DATE])[CODE].count()

            ensemble_long_numbers[index * step] = ensemble_long_number
            ensemble_short_numbers[index * step] = ensemble_short_number
            ensemble_cumulative_returns[index * step] = ensemble_portfolio.get_returns(weighted=WEIGHT, cumulative=True)

        # Fill nan
        ensemble_long_numbers.fillna(0, inplace=True)
        ensemble_short_numbers.fillna(0, inplace=True)
        ensemble_cumulative_returns.fillna(method='ffill', inplace=True)
        ensemble_cumulative_returns.fillna(0, inplace=True)

        print(ensemble_portfolios[-1].outcome(weighted=WEIGHT, show_plot=True))
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

        # Company number
        ensemble_long_numbers.plot(ax=axes[0], colormap='Blues')
        if not long_only:
            ensemble_short_numbers.plot(ax=axes[0], colormap='Oranges')
        axes[0].set_title('{}:{}, Top {}-quantile'.format(method.title(), model_name, quantile))
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('# of companies')
        axes[0].legend(loc='upper left')

        # Cumulative return
        ensemble_cumulative_returns.plot(ax=axes[1], colormap='Blues')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Return')
        axes[1].legend(loc='upper left')

        plt.savefig('summary/' + result_file_name + '.png')
        fig.show()

    return ensemble_summary, ensemble_portfolios


# noinspection PyPep8Naming
def compare_ensemble(methods, models, quantiles, adaptive_outcomes, start_number: int, end_number: int, step: int,
                     long_only: bool, decay: float, to_csv: bool, show_plot: bool):
    file_names = []
    CAGRs = []
    GAGR_rank_correlations = []
    CAGR_rank_p_values = []
    IRs = []
    IR_rank_correlations = []
    IR_rank_p_values = []
    SRs = []
    SR_rank_correlations = []
    SR_rank_p_values = []
    MDDs = []
    alphas = []
    alpha_p_values = []
    alpha_rank_correlations = []
    alpha_rank_p_values = []
    betas = []
    kospi_larges = []
    kospi_middles = []
    kospi_smalls = []
    kosdaq_larges = []
    kosdaq_middles = []
    kosdaq_smalls = []

    firms = Portfolio(
        include_holding=True, include_finance=True, include_managed=True, include_suspended=True
    ).loc[:, [DATE, CODE, MKTCAP, EXCHANGE]]
    firms[DATE] = pd.to_datetime(firms[DATE])

    firms[RANK] = firms[[DATE, EXCHANGE, MKTCAP]].groupby([DATE, EXCHANGE]).rank(ascending=False)
    firms[KOSPI_LARGE] = firms.apply(
        lambda row: 1 if (row[EXCHANGE] == '유가증권시장') and (row[RANK] <= 100) else 0, axis=1)
    firms[KOSPI_MIDDLE] = firms.apply(
        lambda row: 1 if (row[EXCHANGE] == '유가증권시장') and (100 < row[RANK] <= 300) else 0, axis=1)
    firms[KOSPI_SMALL] = firms.apply(
        lambda row: 1 if (row[EXCHANGE] == '유가증권시장') and (300 < row[RANK]) else 0, axis=1)
    firms[KOSDAQ_LARGE] = firms.apply(
        lambda row: 1 if (row[EXCHANGE] == '코스닥') and (row[RANK] <= 100) else 0, axis=1)
    firms[KOSDAQ_MIDDLE] = firms.apply(
        lambda row: 1 if (row[EXCHANGE] == '코스닥') and (100 < row[RANK] <= 300) else 0, axis=1)
    firms[KOSDAQ_SMALL] = firms.apply(
        lambda row: 1 if (row[EXCHANGE] == '코스닥') and (300 < row[RANK]) else 0, axis=1)

    firms = firms.loc[
            :, [DATE, CODE, KOSPI_LARGE, KOSPI_MIDDLE, KOSPI_SMALL, KOSDAQ_LARGE, KOSDAQ_MIDDLE, KOSDAQ_SMALL]
            ]
    with tqdm(total=len(methods) * len(quantiles) * len(models) * len(adaptive_outcomes)) as pbar:
        for method in methods:
            for quantile in quantiles:
                for model in models:
                    for adaptive_outcome in adaptive_outcomes:
                        ensemble_summary, ensemble_portfolios = get_ensemble(
                            method, adaptive_outcome, model, start_number=start_number, end_number=end_number,
                            step=step,
                            long_only=long_only, quantile=quantile, decay=decay, show_plot=show_plot
                        )

                        if ensemble_summary is None and ensemble_portfolios is None:
                            continue

                        file_names.append(_get_file_name(method, model, quantile))

                        CAGRs.append(ensemble_summary[CAGR].values[-1])
                        CAGR_rankIC = spearmanr(ensemble_summary[CAGR].values, ensemble_summary[CAGR].index)
                        GAGR_rank_correlations.append(CAGR_rankIC[0])
                        CAGR_rank_p_values.append(CAGR_rankIC[1])

                        IRs.append(ensemble_summary[IR].values[-1])
                        IR_rankIC = spearmanr(ensemble_summary[IR].values, ensemble_summary[IR].index)
                        IR_rank_correlations.append(IR_rankIC[0])
                        IR_rank_p_values.append(IR_rankIC[1])

                        SRs.append(ensemble_summary[SR].values[-1])
                        SR_rankIC = spearmanr(ensemble_summary[SR].values, ensemble_summary[SR].index)
                        SR_rank_correlations.append(SR_rankIC[0])
                        SR_rank_p_values.append(SR_rankIC[1])

                        MDDs.append(ensemble_summary[MDD].values[-1])

                        alphas.append(ensemble_summary[FAMA_FRENCH_ALPHA].values[-1])
                        alpha_p_values.append(ensemble_summary[FAMA_FRENCH_ALPHA_P_VALUE].values[-1])
                        alpha_rankIC = spearmanr(ensemble_summary[FAMA_FRENCH_ALPHA].values,
                                                 ensemble_summary[FAMA_FRENCH_ALPHA].index)
                        alpha_rank_correlations.append(alpha_rankIC[0])
                        alpha_rank_p_values.append(alpha_rankIC[1])
                        betas.append(ensemble_summary[FAMA_FRENCH_BETA].values[-1])

                        if long_only:
                            # Calculate a composition of the ensemble portfolio
                            # when the portfolio is a long-only portfolio.
                            ensemble_portfolio = pd.merge(ensemble_portfolios[-1], firms, on=[DATE, CODE])
                            ensemble_portfolio_count = ensemble_portfolio[[DATE, CODE]].groupby(DATE).count()
                            ensemble_portfolio_count.rename(columns={CODE: COUNT}, inplace=True)
                            ensemble_portfolio_sum = ensemble_portfolio[[
                                DATE, KOSPI_LARGE, KOSPI_MIDDLE, KOSPI_SMALL, KOSDAQ_LARGE, KOSDAQ_MIDDLE, KOSDAQ_SMALL
                            ]].groupby(DATE).sum()
                            ensemble_portfolio_ratio = pd.merge(ensemble_portfolio_sum, ensemble_portfolio_count,
                                                                on=DATE)
                            ensemble_portfolio_ratio[KOSPI_LARGE] \
                                = ensemble_portfolio_ratio[KOSPI_LARGE] / ensemble_portfolio_ratio[COUNT]
                            ensemble_portfolio_ratio[KOSPI_MIDDLE] \
                                = ensemble_portfolio_ratio[KOSPI_MIDDLE] / ensemble_portfolio_ratio[COUNT]
                            ensemble_portfolio_ratio[KOSPI_SMALL] \
                                = ensemble_portfolio_ratio[KOSPI_SMALL] / ensemble_portfolio_ratio[COUNT]
                            ensemble_portfolio_ratio[KOSDAQ_LARGE] \
                                = ensemble_portfolio_ratio[KOSDAQ_LARGE] / ensemble_portfolio_ratio[COUNT]
                            ensemble_portfolio_ratio[KOSDAQ_MIDDLE] \
                                = ensemble_portfolio_ratio[KOSDAQ_MIDDLE] / ensemble_portfolio_ratio[COUNT]
                            ensemble_portfolio_ratio[KOSDAQ_SMALL] \
                                = ensemble_portfolio_ratio[KOSDAQ_SMALL] / ensemble_portfolio_ratio[COUNT]
                            kospi_larges.append(ensemble_portfolio_ratio[KOSPI_LARGE].mean())
                            kospi_middles.append(ensemble_portfolio_ratio[KOSPI_MIDDLE].mean())
                            kospi_smalls.append(ensemble_portfolio_ratio[KOSPI_SMALL].mean())
                            kosdaq_larges.append(ensemble_portfolio_ratio[KOSDAQ_LARGE].mean())
                            kosdaq_middles.append(ensemble_portfolio_ratio[KOSDAQ_MIDDLE].mean())
                            kosdaq_smalls.append(ensemble_portfolio_ratio[KOSDAQ_SMALL].mean())
                        else:
                            kospi_larges.append(0.0)
                            kospi_middles.append(0.0)
                            kospi_smalls.append(0.0)
                            kosdaq_larges.append(0.0)
                            kosdaq_middles.append(0.0)
                            kosdaq_smalls.append(0.0)

                        pbar.update()

    comparison_result = pd.DataFrame(data={
        'Model': file_names,
        'CAGR': CAGRs,
        'CAGR RC': GAGR_rank_correlations,
        'CAGR RC p-value': CAGR_rank_p_values,
        'IR': IRs,
        'IR RC': IR_rank_correlations,
        'IR RC p-value': IR_rank_p_values,
        'SR': SRs,
        'SR RC': SR_rank_correlations,
        'SR RC p-value': SR_rank_p_values,
        'FF alpha': alphas,
        'FF alpha p-value': alpha_p_values,
        'FF alpha RC': alpha_rank_correlations,
        'FF alpha RC p-value': alpha_rank_p_values,
        'FF betas': betas,
        'MDD': MDDs,
        'KOSPI Large': kospi_larges,
        'KOSPI Middle': kospi_middles,
        'KOSPI Small': kospi_smalls,
        'KOSDAQ Large': kosdaq_larges,
        'KOSDAQ Middle': kosdaq_middles,
        'KOSDAQ Small': kosdaq_smalls,
    })

    if to_csv:
        comparison_result.to_csv('summary/comparison_result.csv', index=False)

    return comparison_result


if __name__ == '__main__':
    models = [
        'DNN8_3-value_size_momentum_quality-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-momentum-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-momentum_quality-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-quality-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-size-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-size_momentum-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-size_momentum_quality-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-size_quality-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-value-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-value_momentum-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-value_momentum_quality-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-value_quality-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-value_size-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-value_size_momentum-tahn-zeros-lecun_normal-none-0.5',
        'DNN8_3-value_size_quality-tahn-zeros-lecun_normal-none-0.5',
    ]
    methods = [
        PORTFOLIO_BLEND,
        SIGNAL_BLEND,
    ]
    quantiles = [
        3,
        5
    ]
    adaptive_outcomes = [
        NONE,
        # SR,
        # IR,
        # CAGR
    ]
    compare_ensemble(methods, models, quantiles, adaptive_outcomes, start_number=0, end_number=9, step=1,
                     long_only=False, decay=0.9, to_csv=True, show_plot=True)
