#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 02:18:30 2020

@author: alejandro
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import bs4 as bs
import requests


plt.style.use('fivethirtyeight')

# Scrap sp500 tickers
def save_sp500_tickers():

    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'html')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        if not '.' in ticker:
            tickers.append(ticker.replace('\n',''))
        
    return tickers

tickers = save_sp500_tickers()

prices = yf.download(tickers, start='2020-01-01')['Adj Close']

rs = prices.apply(np.log).diff(1)
rs.drop(rs.head(1).index,inplace=True)
rs.drop(rs.tail(1).index,inplace=True)

rs.plot(legend=0, figsize=(10,6), grid=True, title='Retornos diarios activos S&P500')
plt.tight_layout()
# plt.savefig('tmp.png')


(rs.cumsum().apply(np.exp)).plot(legend=0, figsize=(10,6), grid=True, title='Retornos acumulados activos S&P500')
plt.tight_layout()
# plt.savefig('tmp.png')



from sklearn.decomposition import PCA

pca = PCA(1).fit(rs.fillna(0))
pca.singular_values_[0]
pca.components_[0]
pca.explained_variance_[0]


pca_1 = PCA().fit(rs.fillna(0))
pca_1.explained_variance_[0]
pca_1.singular_values_[0]
pca_1.components_[0]
pca_1.explained_variance_[1]
pca_1.singular_values_[1]
pca_1.components_[1]
pca_1.explained_variance_[2]
pca_1.singular_values_[2]
pca_1.components_[2]

import seaborn as sns

covMatrix = pca_1.get_covariance()

sns.heatmap(covMatrix, vmax=.3, square=True)

mask = np.zeros_like(covMatrix)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.heatmap(covMatrix, mask=mask, vmax=.3, square=True, 
                      cmap = 'YlGnBu')


pc1 = pd.Series(index=rs.columns, data=pca.components_[0])

pc1.plot(figsize=(10,6), xticks=[], grid=True, title='Primer Componente Principal S&P500')
plt.tight_layout()
# plt.savefig('tmp.png')

weights = abs(pc1)/sum(abs(pc1))
myrs = (weights*rs).sum(1)
myrs.cumsum().apply(np.exp).plot()


prices = yf.download(['SPY'], start='2020-01-01')['Adj Close']


rs_df = pd.concat([myrs, prices.apply(np.log).diff(1)], 1)
rs_df.columns = ["PCA Portfolio", "S&P500"]

rs_df.dropna().cumsum().apply(np.exp).plot(subplots=True, figsize=(10,6), grid=True, linewidth=3);
plt.tight_layout()
# plt.savefig('tmp.png')


fig, ax = plt.subplots(2,1, figsize=(10,6))
pc1.nsmallest(10).plot.bar(ax=ax[0], color='green', grid=True, title='Stocks with Most Negative PCA Weights')
pc1.nlargest(10).plot.bar(ax=ax[1], color='blue', grid=True, title='Stocks with Least Negative PCA Weights')
plt.tight_layout()
# plt.savefig('tmp.png')

# ws = [-1,]*10+[1,]*10
# myrs = (rs[list(pc1.nsmallest(10).index)+list(pc1.nlargest(10).index)]*ws).mean(1)
myrs = rs[pc1.nlargest(10).index].mean(1)
myrs.cumsum().apply(np.exp).plot(figsize=(15,5), grid=True, linewidth=3, title='PCA Portfolio vs. S&P500')
prices['2020':].apply(np.log).diff(1).cumsum().apply(np.exp).plot(figsize=(10,6), grid=True, linewidth=3)
plt.legend(['PCA Selection', 'S&P500'])

plt.tight_layout()
plt.savefig('tmp.png')