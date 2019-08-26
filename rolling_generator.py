import pandas as pd
from ksif import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

_msf = pd.read_csv('E:/pjs/ksif/data/upload2dropbox/final_msf.csv')

FOREIGN_OWNERSHIP_RATIO = 'foreign'

rolling_columns = [
    MKTCAP,
    E_P, B_P, S_P, C_P, OP_P, GP_P, ROA, ROE, QROA, QROE, GP_A, ROIC, GP_S, SALESQOQ, GPQOQ, ROAQOQ,
    MOM6, MOM12, BETA_1D, VOL_5M, LIQ_RATIO, EQUITY_RATIO, DEBT_RATIO, FOREIGN_OWNERSHIP_RATIO
]

control_list = [
    TERM_SPREAD_KOR, TERM_SPREAD_US, CREDIT_SPREAD_KOR, LOG_USD2KRW, LOG_CHY2KRW, LOG_EURO2KRW,
    TED_SPREAD, LOG_NYSE, LOG_NASDAQ, LOG_OIL, KRX_SECTOR]

msf = _msf[[CODE, DATE, RET_1] + rolling_columns + control_list].copy(deep=True)
msf = msf.loc[msf[MKTCAP] > 10000000000, :]

# KRX_SECTOR가 존재하지 않는 데이터 제거
msf.dropna(subset=[KRX_SECTOR], inplace=True)
msf = msf.sort_values(by=[CODE, DATE]).reset_index(drop=True)
msf.fillna(0, inplace=True)

# ss = msf[msf['code']=='A005930'].copy(deep=True)
sample = msf[(msf['code']=='A005930') | (msf['code']=='A000660')].copy(deep=True)

# df_list = []
# for i in range(13):
#     rolled_df = sample[rolling_columns].shift(i)
#     colname = [name+'_t-{}'.format(i) for name in rolling_columns]
#     rolled_df.columns = colname
#     df_list.append(rolled_df)
# sample['check_code'] = sample['code'].shift(12)
# _final_df = pd.concat([sample]+df_list, axis=1)
# final_df = _final_df[_final_df['code']==_final_df['check_code']]

df_list = []
for i in range(13):
    if i>0:
        rolled_df = msf[rolling_columns].shift(i)
    else:
        rolled_df = msf[rolling_columns]
    colname = [name+'_t-{}'.format(i) for name in rolling_columns]
    rolled_df.columns = colname
    df_list.append(rolled_df)
msf['check_code'] = msf['code'].shift(12)
msf_head = msf.drop(columns=rolling_columns).copy(deep=True)
_final_df = pd.concat([msf_head]+df_list, axis=1)
final_df = _final_df[_final_df['code']==_final_df['check_code']]
final_df.drop(columns='check_code', inplace=True)


# sector를 숫자로 나타냄
label_encoder = LabelEncoder()
labeled_sector = label_encoder.fit_transform(final_df[KRX_SECTOR])
krx_sectors = label_encoder.classes_
# 숫자로 나타낸 것을 모스부호로 표현
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_encoded_sector = one_hot_encoder.fit_transform(labeled_sector.reshape(len(labeled_sector), 1))
# 기존 데이터에 붙히기
df_one_hot_encoded_sector = pd.DataFrame(one_hot_encoded_sector, columns=krx_sectors).reset_index(drop=True)
final_df[krx_sectors] = df_one_hot_encoded_sector
final_df.drop(columns=KRX_SECTOR,inplace=True)
final_df.dropna(inplace=True)

final_df.to_csv('./data/final_df.csv')

ss = final_df[final_df[CODE]=='A005930'].copy(deep=True)