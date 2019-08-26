# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018-10-05
"""

# dropbox에 데이터 올릴 동안은 업로드 중이라는 임시 flag를 생성해두고, FML에서 predict해야할 때는 그것이 있을 때는 계속 한 3분에 한번씩? access하게 시켰다가,
# 업로드 되면, 그 임시 flag 삭제하고, 데이터 다운 받아서 쭉 하면 되겠지.
# 그러면, ksif로 전처리 올려두고,
# FML로 데이터 다운 받아서 학습시키고, predict하면 끝!
# 끝!