# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 11. 19.
"""
import pandas as pd
from ksif.core.columns import *

all_count = pd.read_csv('data/all.csv').groupby(by=[DATE])[CODE].count()
kospi_count = pd.read_csv('data/kospi.csv').groupby(by=[DATE])[CODE].count()
kosdaq_count = pd.read_csv('data/kosdaq.csv').groupby(by=[DATE])[CODE].count()

result = pd.DataFrame(data={
    'all': all_count,
    'kospi': kospi_count,
    'kosdaq': kosdaq_count
})

result.to_csv('universe_number/universe_number.csv')
