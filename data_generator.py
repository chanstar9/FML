# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018-09-21
"""
import numpy as np
import pandas as pd
from ksif import Portfolio
from ksif.core.columns import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tqdm import tqdm

# DATA_SET
ALL = 'all'
FILTER = 'filter'
BOLLINGER = 'bollinger'
SECTOR = 'sector'

TRADING_CAPITAL = 'trading_capital'

START_DATE = '2004-05-31'
USED_PAST_MONTHS = 12  # At a time, use past 12 months data and current month data.


def get_data_set(portfolio, rolling_columns, dummy_columns=None, return_y=True, apply_scaling=True):
    if return_y:
        result_columns = [DATE, CODE, RET_1]
    else:
        result_columns = [DATE, CODE]
    data_set = portfolio.sort_values(by=[CODE, DATE]).reset_index(drop=True)

    # MinMaxScale
    if apply_scaling:
        # Return
        data_set[RET_1] = data_set.groupby(by=[DATE])[RET_1].apply(
            lambda x: (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)))

        # Rolling columns
        for i in rolling_columns:
            data_set[i] = data_set.groupby(by=[DATE])[i].apply(
                lambda x: (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)))

    for column in tqdm(rolling_columns):
        for i in range(0, USED_PAST_MONTHS + 1):
            column_i = column + '_t-{}'.format(i)
            result_columns.append(column_i)
            data_set[column_i] = data_set.groupby(by=[CODE]).apply(lambda x: x[column].shift(i)).reset_index(drop=True)

    if dummy_columns is not None:
        result_columns.extend(dummy_columns)

    data_set = data_set[result_columns]
    data_set = data_set.dropna().reset_index(drop=True)

    return data_set


def save_data(portfolio: Portfolio, data_name: str, rolling_columns: list, dummy_columns: list = None,
              filtering_dataframe=None):
    print("Start saving {}...".format(data_name))

    portfolio = portfolio.sort_values(by=[CODE, DATE]).reset_index(drop=True)

    # old data
    # RET_1이 존재하지 않는 마지막 달 제거
    old_portfolio = portfolio.loc[~pd.isna(portfolio[RET_1]), :]
    old_set = get_data_set(old_portfolio, rolling_columns, dummy_columns)

    # recent data
    recent_set = get_data_set(portfolio, rolling_columns, dummy_columns, return_y=False)
    # 마지막 달만 사용
    last_month = np.sort(recent_set[DATE].unique())[-1]
    recent_set = recent_set.loc[recent_set[DATE] == last_month, :]

    if isinstance(filtering_dataframe, pd.DataFrame) and not filtering_dataframe.empty:
        filtering_dataframe = filtering_dataframe[[DATE, CODE]]
        old_set = pd.merge(old_set, filtering_dataframe, on=[DATE, CODE])
        recent_set = pd.merge(old_set, filtering_dataframe, on=[DATE, CODE])

    old_set.to_dataframe().to_hdf('data/{}.h5'.format(data_name), key='df', format='table', mode='w')
    recent_set.to_dataframe().to_hdf('data/{}_recent.h5'.format(data_name), key='df', format='table', mode='w')


def save_all():
    rolling_columns = [E_P, B_P, S_P, C_P, OP_P, GP_P, ROA, ROE, QROA, QROE, GP_A, ROIC, GP_S, SALESQOQ, GPQOQ, ROAQOQ,
                       MOM6, MOM12, BETA_1D, VOL_5M, LIQ_RATIO, EQUITY_RATIO, DEBT_RATIO, FOREIGN_OWNERSHIP_RATIO]
    portfolio = Portfolio()
    # 최소 시가총액 100억
    portfolio = portfolio.loc[portfolio[MKTCAP] > 10000000000, :]
    # 월 거래액 10억 이상
    portfolio[TRADING_CAPITAL] = portfolio[TRADING_VOLUME_RATIO] * portfolio[MKTCAP]

    save_data(portfolio, ALL, rolling_columns)


def save_filter():
    rolling_columns = [E_P, B_P, S_P, C_P, OP_P, GP_P, ROA, ROE, QROA, QROE, GP_A, ROIC, GP_S, SALESQOQ, GPQOQ, ROAQOQ,
                       MOM6, MOM12, BETA_1D, VOL_5M, LIQ_RATIO, EQUITY_RATIO, DEBT_RATIO, FOREIGN_OWNERSHIP_RATIO]
    portfolio = Portfolio()
    # 최소 시가총액 100억
    portfolio = portfolio.loc[portfolio[MKTCAP] > 10000000000, :]
    # 월 거래액 10억 이상
    portfolio[TRADING_CAPITAL] = portfolio[TRADING_VOLUME_RATIO] * portfolio[MKTCAP]
    # 2 < PER < 10.0 (http://pluspower.tistory.com/9)
    portfolio = portfolio.loc[(portfolio[PER] < 10) & (portfolio[PER] > 2)]
    # 0.2 < PBR < 1.0
    portfolio = portfolio.loc[(portfolio[PBR] < 1) & (portfolio[PBR] > 0.2)]
    # 2 < PCR < 8
    portfolio = portfolio.loc[(portfolio[PCR] < 8) & (portfolio[PCR] > 2)]
    # 0 < PSR < 0.8
    portfolio = portfolio.loc[portfolio[PSR] < 0.8]

    save_data(portfolio, FILTER, rolling_columns)


def save_bollinger():
    rolling_columns = [E_P, B_P, S_P, C_P, OP_P, GP_P, ROA, ROE, QROA, QROE, GP_A, ROIC, GP_S, SALESQOQ, GPQOQ, ROAQOQ,
                       MOM6, MOM12, BETA_1D, VOL_5M, LIQ_RATIO, EQUITY_RATIO, DEBT_RATIO, FOREIGN_OWNERSHIP_RATIO]
    portfolio = Portfolio()
    # 최소 시가총액 100억
    portfolio = portfolio.loc[portfolio[MKTCAP] > 10000000000, :]
    # 월 거래액 10억 이상
    portfolio[TRADING_CAPITAL] = portfolio[TRADING_VOLUME_RATIO] * portfolio[MKTCAP]

    # Bollinger
    portfolio = portfolio.sort_values(by=[CODE, DATE]).reset_index(drop=True)
    portfolio['mean'] = portfolio.groupby(CODE)[ENDP].rolling(20).mean().reset_index(drop=True)
    portfolio['std'] = portfolio.groupby(CODE)[ENDP].rolling(20).std().reset_index(drop=True)
    portfolio[BOLLINGER] = portfolio['mean'] - 2 * portfolio['std']
    bollingers = portfolio.loc[portfolio[ENDP] < portfolio[BOLLINGER], [DATE, CODE]]

    save_data(portfolio, BOLLINGER, rolling_columns, filtering_dataframe=bollingers)


def save_sector():
    columns = [DATE, CODE, RET_1]
    rolling_columns = [E_P, B_P, S_P, C_P, OP_P, GP_P, ROA, ROE, QROA, QROE, GP_A, ROIC, GP_S, SALESQOQ, GPQOQ, ROAQOQ,
                       MOM6, MOM12, BETA_1D, VOL_5M, LIQ_RATIO, EQUITY_RATIO, DEBT_RATIO, FOREIGN_OWNERSHIP_RATIO]
    columns.extend(rolling_columns)
    portfolio = Portfolio()
    # 최소 시가총액 100억
    portfolio = portfolio.loc[portfolio[MKTCAP] > 10000000000, :]
    # 월 거래액 10억 이상
    portfolio[TRADING_CAPITAL] = portfolio[TRADING_VOLUME_RATIO] * portfolio[MKTCAP]

    # KRX_SECTOR가 존재하지 않는 데이터 제거
    portfolio.dropna(subset=[KRX_SECTOR], inplace=True)
    portfolio = portfolio.sort_values(by=[CODE, DATE]).reset_index(drop=True)

    # sector를 숫자로 나타냄
    label_encoder = LabelEncoder()
    labeled_sector = label_encoder.fit_transform(portfolio[KRX_SECTOR])
    krx_sectors = label_encoder.classes_
    # 숫자로 나타낸 것을 모스부호로 표현
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoded_sector = one_hot_encoder.fit_transform(labeled_sector.reshape(len(labeled_sector), 1))
    # 기존 데이터에 붙히기
    df_one_hot_encoded_sector = pd.DataFrame(one_hot_encoded_sector, columns=krx_sectors).reset_index(drop=True)
    portfolio[krx_sectors] = df_one_hot_encoded_sector

    save_data(portfolio, SECTOR, rolling_columns, krx_sectors)


if __name__ == '__main__':
    # save_all()
    save_filter()
    save_bollinger()
    save_sector()
