# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018-12-20
"""
import pandas as pd
from ksif import Portfolio
from ksif.core.columns import *

pf = Portfolio()

total_periodic_integrity = pd.DataFrame()
for factor in COMPANY_FACTORS:
    integrative_factors = pf.loc[~pd.isna(pf[factor]), [DATE, factor]]
    all_factors = pf.loc[:, [DATE, factor]]
    integrative_counts = integrative_factors.groupby(DATE).count()
    all_counts = all_factors.groupby(DATE).count()
    factor_periodic_integrity = integrative_counts / all_counts
    factor_periodic_integrity.fillna(0, inplace=True)
    total_periodic_integrity = pd.concat([total_periodic_integrity, factor_periodic_integrity], axis=1, sort=True)

total_periodic_integrity.to_csv('periodic_integrity/periodic_integrity.csv', encoding='utf-8')
