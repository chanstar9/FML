# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018-09-21
"""
import pandas as pd
from ksif import Portfolio
from ksif.core.columns import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# DATA_SET
ALL = 'all'
FILTER = 'filter'
BOLLINGER = 'bollinger'
SECTOR = 'sector'

START_DATE = '2007-04-30'
USED_PAST_MONTHS = 12  # At a time, use past 12 months data and current month data.


def get_data_set(portfolio, rolling_columns, dummy_columns=None):
    result_columns = [DATE, CODE, RET_1]
    data_set = portfolio.sort_values(by=[CODE, DATE]).reset_index(drop=True)
    for column in rolling_columns:
        for i in range(0, USED_PAST_MONTHS + 1):
            column_i = column + '_t-{}'.format(i)
            result_columns.append(column_i)
            data_set[column_i] = data_set.groupby(by=[CODE]).apply(lambda x: x[column].shift(i)).reset_index(drop=True)

    if dummy_columns is not None:
        result_columns.extend(dummy_columns)

    data_set = data_set[result_columns]
    data_set = data_set.dropna().reset_index(drop=True)

    return data_set


def save_all():
    rolling_columns = [E_P, B_P, S_P, C_P, OP_P, GP_P, ROA, ROE, QROA, QROE, GP_A, ROIC, GP_S, SALESQOQ, GPQOQ, ROAQOQ,
                       MOM6, MOM12, BETA_1D, VOL_5M, LIQ_RATIO, EQUITY_RATIO, DEBT_RATIO, FOREIGN_OWNERSHIP_RATIO]
    all_portfolio = Portfolio()
    # 최소 시가총액 100억
    all_portfolio = all_portfolio.loc[all_portfolio[MKTCAP] > 10000000000, :]
    # RET_1이 존재하지 않는 마지막 달 제거
    all_portfolio = all_portfolio.loc[~pd.isna(all_portfolio[RET_1]), :]
    all_portfolio = all_portfolio.sort_values(by=[CODE, DATE]).reset_index(drop=True)

    all_set = get_data_set(all_portfolio, rolling_columns)
    all_set.to_csv('data/all.csv', index=False)


def save_filter():
    rolling_columns = [E_P, B_P, S_P, C_P, OP_P, GP_P, ROA, ROE, QROA, QROE, GP_A, ROIC, GP_S, SALESQOQ, GPQOQ, ROAQOQ,
                       MOM6, MOM12, BETA_1D, VOL_5M, LIQ_RATIO, EQUITY_RATIO, DEBT_RATIO, FOREIGN_OWNERSHIP_RATIO]
    all_portfolio = Portfolio()
    # 최소 시가총액 100억
    all_portfolio = all_portfolio.loc[all_portfolio[MKTCAP] > 10000000000, :]
    # RET_1이 존재하지 않는 마지막 달 제거
    all_portfolio = all_portfolio.loc[~pd.isna(all_portfolio[RET_1]), :]
    all_portfolio = all_portfolio.sort_values(by=[CODE, DATE]).reset_index(drop=True)

    # 2 < PER < 10.0 (http://pluspower.tistory.com/9)
    all_portfolio = all_portfolio.loc[(all_portfolio[PER] < 10) & (all_portfolio[PER] > 2)]
    # 0.2 < PBR < 1.0
    all_portfolio = all_portfolio.loc[(all_portfolio[PBR] < 1) & (all_portfolio[PBR] > 0.2)]
    # 2 < PCR < 8
    all_portfolio = all_portfolio.loc[(all_portfolio[PCR] < 8) & (all_portfolio[PCR] > 2)]
    # 0 < PSR < 0.8
    all_portfolio = all_portfolio.loc[all_portfolio[PSR] < 0.8]

    all_set = get_data_set(all_portfolio, rolling_columns)
    all_set.to_csv('data/filter.csv', index=False)


def save_bollinger():
    rolling_columns = [E_P, B_P, S_P, C_P, OP_P, GP_P, ROA, ROE, QROA, QROE, GP_A, ROIC, GP_S, SALESQOQ, GPQOQ, ROAQOQ,
                       MOM6, MOM12, BETA_1D, VOL_5M, LIQ_RATIO, EQUITY_RATIO, DEBT_RATIO, FOREIGN_OWNERSHIP_RATIO]
    all_portfolio = Portfolio()
    # 최소 시가총액 100억
    all_portfolio = all_portfolio.loc[all_portfolio[MKTCAP] > 10000000000, :]
    # RET_1이 존재하지 않는 마지막 달 제거
    all_portfolio = all_portfolio.loc[~pd.isna(all_portfolio[RET_1]), :]
    all_portfolio = all_portfolio.sort_values(by=[CODE, DATE]).reset_index(drop=True)

    # Bollinger
    all_portfolio = all_portfolio.sort_values(by=[CODE, DATE]).reset_index(drop=True)
    all_portfolio['mean'] = all_portfolio.groupby(CODE)[ENDP].rolling(20).mean().reset_index(drop=True)
    all_portfolio['std'] = all_portfolio.groupby(CODE)[ENDP].rolling(20).std().reset_index(drop=True)
    all_portfolio[BOLLINGER] = all_portfolio['mean'] - 2 * all_portfolio['std']
    bollingers = all_portfolio.loc[all_portfolio[ENDP] < all_portfolio[BOLLINGER], [DATE, CODE]]

    all_set = get_data_set(all_portfolio, rolling_columns)
    all_set = pd.merge(all_set, bollingers, on=[DATE, CODE])
    all_set.to_csv('data/bollinger.csv', index=False)


def save_sector():
    columns = [DATE, CODE, RET_1]
    rolling_columns = [E_P, B_P, S_P, C_P, OP_P, GP_P, ROA, ROE, QROA, QROE, GP_A, ROIC, GP_S, SALESQOQ, GPQOQ, ROAQOQ,
                       MOM6, MOM12, BETA_1D, VOL_5M, LIQ_RATIO, EQUITY_RATIO, DEBT_RATIO, FOREIGN_OWNERSHIP_RATIO]
    columns.extend(rolling_columns)
    all_portfolio = Portfolio()
    # 최소 시가총액 100억
    all_portfolio = all_portfolio.loc[all_portfolio[MKTCAP] > 10000000000, :]
    # RET_1이 존재하지 않는 마지막 달 제거
    all_portfolio = all_portfolio.loc[~pd.isna(all_portfolio[RET_1]), :]
    # KRX_SECTOR가 존재하지 않는 데이터 제거
    all_portfolio.dropna(subset=[KRX_SECTOR], inplace=True)
    all_portfolio = all_portfolio.sort_values(by=[CODE, DATE]).reset_index(drop=True)

    # sector를 숫자로 나타냄
    label_encoder = LabelEncoder()
    labeled_sector = label_encoder.fit_transform(all_portfolio[KRX_SECTOR])
    krx_sectors = label_encoder.classes_
    # 숫자로 나타낸 것을 모스부호로 표현
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoded_sector = one_hot_encoder.fit_transform(labeled_sector.reshape(len(labeled_sector), 1))
    # 기존 데이터에 붙히기
    df_one_hot_encoded_sector = pd.DataFrame(one_hot_encoded_sector, columns=krx_sectors).reset_index(drop=True)
    all_portfolio[krx_sectors] = df_one_hot_encoded_sector
    # 데이터 생성하기
    all_set = get_data_set(all_portfolio, rolling_columns, dummy_columns=krx_sectors)
    all_set.to_csv('data/sector.csv', index=False)


if __name__ == '__main__':
    save_all()
    save_filter()
    save_bollinger()
    save_sector()
