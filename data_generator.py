# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018-09-21
"""
import pandas as pd
from ksif import Portfolio
from ksif.core.columns import *
from sklearn.preprocessing import MinMaxScaler

START_DATE = '2007-04-30'
USED_PAST_MONTHS = 12  # At a time, use past 12 months data and current month data.

scaler = MinMaxScaler()


def get_data_set(portfolio, rolling_columns):
    result_columns = [DATE, CODE, RET_1]
    rolled_columns = []
    data_set = portfolio.reset_index(drop=True)
    for column in rolling_columns:
        for i in range(0, USED_PAST_MONTHS + 1):
            column_i = column + '_t-{}'.format(i)
            result_columns.append(column_i)
            rolled_columns.append(column_i)
            data_set[column_i] = data_set.groupby(by=[CODE]).apply(lambda x: x[column].shift(i)).reset_index(drop=True)
    data_set = data_set[result_columns]
    data_set = data_set.dropna().reset_index(drop=True)
    data_set[rolled_columns] = scaler.fit_transform(data_set[rolled_columns])

    return data_set


def save_all():
    fixed_columns = [DATE, CODE, RET_1]
    rolling_columns = [E_P, B_P, S_P, C_P, OP_P, GP_P, ROA, ROE, QROA, QROE, GP_A, ROIC, GP_S, SALESQOQ, GPQOQ, ROAQOQ,
                       MOM6, MOM12, BETA_1D, VOL_5M, LIQ_RATIO, EQUITY_RATIO, DEBT_RATIO, FOREIGN_OWNERSHIP_RATIO]
    fixed_columns.extend(rolling_columns)
    all_portfolio = Portfolio(start_date=START_DATE)
    # 최소 시가총액 100억
    all_portfolio = all_portfolio.loc[all_portfolio[MKTCAP] > 10000000000, :]
    # RET_1이 존재하지 않는 마지막 달 제거
    all_portfolio = all_portfolio.loc[~pd.isna(all_portfolio[RET_1]), :]
    all_set = get_data_set(all_portfolio, rolling_columns)
    all_set.to_csv('data/all.csv', index=False)


if __name__ == '__main__':
    save_all()
