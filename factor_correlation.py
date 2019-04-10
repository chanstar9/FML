import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ksif import *

if __name__ == '__main__':
    log_mktcap = 'log_mktcap'
    portfolio = Portfolio()
    portfolio[log_mktcap] = np.log(portfolio[MKTCAP])
    value_factors = [E_P, B_P, S_P, C_P, DIVP]
    size_factors = [log_mktcap]
    momentum_factors = [MOM1, MOM12]
    quality_factors = [ROA, ROE, ROIC, S_A, DEBT_RATIO, EQUITY_RATIO, LIQ_RATIO]
    factors = []
    factors.extend(value_factors)
    factors.extend(size_factors)
    factors.extend(momentum_factors)
    factors.extend(quality_factors)
    portfolio = portfolio.dropna(subset=factors)
    corr = portfolio[factors].corr()
    corr.to_csv('correlation.csv')
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='coolwarm')
    plt.savefig('correlation.png')
    plt.show()
