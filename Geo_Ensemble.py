# Geometric Ensemble
# Author: YUN SHIM

# 사용법:
# 1) 스크립트를 실행하기 전에 Geo_Ensemble_S 폴더를 먼저 만듭니다.
# 2) 앙상블할 결과 파일을 Geo_Ensemble_S 폴더에 저장합니다. 본 코드는 100개 미만의 파일들을 누적하여 앙상블해서
# 저장합니다. 결과 파일들의 파일명의 첫 두자리는 숫자로 나타내주세요. ex) 02-blahblah.csv, 14-blahblah.csv
# 3) 실행하면 현재 폴더에 앙상블한 submission 화일들이 생성됩니다. 0_? 는 0부터 ? 까지의 파일을 앙상블한 결과
# 임을 의미합니다.

#%%
import pandas as pd
import numpy as np
import os
#%%
folder = 'Geo_Ensemble_S'
num_files = 30
for ne in range(num_files):
    nf = 0
    for f in os.listdir(folder):
        if ne < int(f[0:2]):
            continue
        ext = os.path.splitext(f)[-1]
        if ext == '.csv':
            s = pd.read_csv(folder+"/"+f)
        else:
            continue
        if len(s.columns) !=7:
            continue
        if nf == 0:
            merged = s[["date", "code", "predict_return_1"]]
        else:
            merged = pd.merge(merged, s[["date", "code", "predict_return_1"]], on=["date", "code"])
        nf += 1

    if nf >= 2:
        pred = 1
        for j in range(nf): pred = pred * np.exp(merged.iloc[:, j+2])
        pred = pred**(1/nf)
    else:
        pred = np.exp(merged.iloc[:, 2])

    dfs = pd.read_csv(folder+"/"+'00_Defaults_for_Geo.csv')

    submit = pd.DataFrame({'date': dfs.date, 'code': merged.code, 'return_1': dfs.return_1, 'pred_return_e': pred,
                       'actual_rank': dfs.actual_rank})

    submit['predict_rank'] = submit.groupby('date')['pred_return_e'].rank(ascending=False)

    t = pd.Timestamp.now()
    # fname = '0_' + str(ne) + "_sub_ES_" + ".csv"
    fname = '0_' + str(ne) + "_sub_ES_" + str(t.month) + str(t.day) + "_" + str(t.hour) + str(t.minute) + ".csv"

    submit.to_csv(fname, index=False)

