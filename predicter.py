# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018-09-21
"""
# noinspection PyUnresolvedReferences
from ensemble import INTERSECTION, GEOMETRIC, ARITHMETIC
from model import *
from settings import *
from bs4 import BeautifulSoup
import requests

NET_INCOME_FILTER = 'net_income_filter'
NET_INCOME_FILTER_SELECTION = 'net_income_filter_selection'

import time
import urllib3

if __name__ == '__main__':

    # memo_in_saving = 'MY_INVEST_PORT3_20191230'
    # memo_in_saving = 'MY_INVEST_PORT4_20191230'

    # for memo_in_saving in ['MY_INVEST_RNN_NET_INCOME_FILTERED_in_training_and_not_selection_with_Quarterly_updated_data_20191202', 'MY_INVEST_RNN_NET_INCOME_FILTERED_in_training_and_selection_with_Quarterly_updated_data_20191202', 'dnn_20191202']:
    for memo_in_saving in ['MY_INVEST_PORT3_20191230', 'MY_INVEST_PORT4_20191230', 'MY_INVEST_PORT5_dnn_20191230']:
        if 'PORT3' in memo_in_saving:
            NI_selection_true_false = False
        elif 'PORT4' in memo_in_saving:
            NI_selection_true_false = True
        else:
            NI_selection_true_false = False

        if 'dnn' in memo_in_saving:
            network_architecture_to_train = DNN8_2
        else:
            network_architecture_to_train = RNN8_2

        time_of_saving = get_forward_predict(
            param={
                DATA_SET: ALL,
                BATCH_SIZE: 256,
                EPOCHS: 100,
                ACTIVATION: LINEAR,
                BIAS_INITIALIZER: HE_UNIFORM,
                KERNEL_INITIALIZER: GLOROT_UNIFORM,
                BIAS_REGULARIZER: NONE,
                HIDDEN_LAYER: network_architecture_to_train,
                DROPOUT: False,
                DROPOUT_RATE: 0.5,
                NET_INCOME_FILTER: True,
                NET_INCOME_FILTER_SELECTION: NI_selection_true_false,
                VERBOSE: 0},
            quantile=40, model_num=10, method=GEOMETRIC
        )

        forward_predict_list = pd.read_csv('forward_predict/forward_predictions_{}.csv'.format(time_of_saving))
        # memo_in_saving = 'exercise_of_checking_whether_loss_decreasing_well'
        # forward_predict_list = pd.read_csv('forward_predict/forward_predictions-GRU-rebal.csv')
        # memo_in_saving = 'net_income_filtered'


        if forward_predict_list.columns[0][0] == 'A':
            first_firm = forward_predict_list.columns.to_frame()
            first_firm.columns = ['code']
            first_firm.index = [forward_predict_list.shape[0]]
            forward_predict_list.columns = ['code']
            forward_predict_list2 = first_firm.append(forward_predict_list)

        for _, code in forward_predict_list2.iterrows():
            code_str = code.values[0][1:]

            req = requests.get('https://finance.naver.com/item/sise.nhn?code={}'.format(code_str))
            html = req.text
            soup = BeautifulSoup(html, 'html.parser')
            cprice = int(soup.find(id='_nowVal').text.replace(',',''))
            company_name = soup.find('div', {'class':'wrap_company'}).text.split('\n')[1]

            market_cap_info = soup.find('div', {'class': 'first'})
            market_cap = market_cap_info.find(id='_market_sum').text.replace('\n', '').replace('\t','')

            market_cap_rank = market_cap_info.find_all('td')[1].text

            per_info = soup.find(id='_per').text
            pbr_info = soup.find(id='_pbr').text
            _dividend_return = soup.find(id='_dvr')
            if _dividend_return is not None:
                dividend_return = _dividend_return.text
            else:
                dividend_return = np.nan

            _per_industry = soup.find('table', {'summary':'동일업종 PER 정보'})
            per_industry = _per_industry.text[:_per_industry.text.find('배')+1].split('\n')[-1][:-1]

            time.sleep(0.01)

            req = requests.get('https://finance.naver.com/item/coinfo.nhn?code={}'.format(code_str))
            html = req.text
            soup = BeautifulSoup(html, 'html.parser')

            iframexx = soup.find_all('iframe')

            http = urllib3.PoolManager()

            for iframe in iframexx:
                if iframe.attrs['id'] == 'coinfo_cp':
                    source_url = iframe.attrs['src']
                    # print(source_url)
                    req_iframe = requests.get(source_url)
                    html_iframe = req_iframe.text
                    soup_iframe= BeautifulSoup(html_iframe, 'html.parser')

            industry_broader = soup_iframe.find(id='pArea').find_all('dt', {'class': 'line-left'})[1].text
            industry_specific = soup_iframe.find(id='pArea').find_all('dt', {'class': 'line-left'})[2].text
            pbr_info = soup_iframe.find(id='pArea').find_all('dt', {'class': 'line-left'})[-2].text
            per_info = soup_iframe.find(id='pArea').find_all('dt', {'class': 'line-left'})[-4].text
            per_industry = soup_iframe.find(id='pArea').find_all('dt', {'class': 'line-left'})[-3].text
            dividend_return = soup_iframe.find(id='pArea').find_all('dt', {'class': 'line-left'})[-1].text


            forward_predict_list2.loc[_, '붙여넣기 용'] = str(code_str)
            forward_predict_list2.loc[_, '회사명'] = company_name
            forward_predict_list2.loc[_, '현재가'] = cprice

            forward_predict_list2.loc[_, '시가총액'] = market_cap
            forward_predict_list2.loc[_, '시가총액 순위'] = market_cap_rank

            forward_predict_list2.loc[_, 'PER'] = per_info.split('PER ')[1]
            forward_predict_list2.loc[_, '산업 PER'] = per_industry.split('업종PER ')[1]
            forward_predict_list2.loc[_, 'PBR'] = pbr_info.split('PBR ')[1]
            forward_predict_list2.loc[_, '배당수익률'] = dividend_return.split('현금배당수익률 ')[1]

            forward_predict_list2.loc[_, '업종 대분류'] = industry_broader
            forward_predict_list2.loc[_, '업종 소분류'] = industry_specific.split('WICS : ')[1]



            time.sleep(0.01)

        forward_predict_list2.set_index('code', inplace=True)
        # time_of_saving = datetime.now().strftime('%Y%m%d-%H%M%S')
        forward_predict_list2.to_excel('forward_predict/forward_prediction_with_info-{}-{}.xlsx'.format(memo_in_saving, time_of_saving))
