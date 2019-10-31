import pandas as pd
import os
from ksif.core.columns import *
df = pd.read_csv('E:/pjs/ksif/data/190831_company.csv')

df.columns
for col in df.columns:
    if 'foreign' in col:
        print(col)