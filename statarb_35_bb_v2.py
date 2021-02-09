# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 09:20:33 2021

@author: Meva
"""

# import libraries and functions
import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress
from scipy.optimize import minimize
from numpy import linalg as LA

# import our own files and reload
import stream_functions
importlib.reload(stream_functions)
import stream_classes
importlib.reload(stream_classes)
import bollinger_bands
importlib.reload(bollinger_bands)

# inputs
backtest = bollinger_bands.backtest()
backtest.ric_long = 'VWS.CO'
backtest.ric_short = 'SGREN.MC'
backtest.data_cut = 0.7
backtest.data_type = 'in-sample' # in-sample out-of-sample

# list_level_1 = [0.5,0.6,0.7,0.8,0.9]
# list_level_2 = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8]

list_level_1 = [np.round(x,2) for x in np.linspace(0.5, 0.75, num=6)]
list_level_2 = [np.round(x,2) for x in np.linspace(1.0, 1.5, num=11)]

mtx_sharpe = np.zeros((len(list_level_1),len(list_level_2)))
mtx_nb_trades = np.zeros((len(list_level_1),len(list_level_2)))

l1 = 0
for level_1 in list_level_1:
    l2 = 0
    for level_2 in list_level_2:
        # parameters to optimise
        backtest.rolling_days = 35
        backtest.level_1 = level_1
        backtest.level_2 = level_2
        # load data
        backtest.load_data()
        # compute indicator
        backtest.compute_indicators()
        # run backtest of trading strategy
        backtest.run_strategy()
        # get results in a dataframe format
        df_strategy = backtest.dataframe_strategy
        # save data in tables
        mtx_sharpe[l1][l2] = backtest.pnl_sharpe
        mtx_nb_trades[l1][l2] = backtest.nb_trades
        l2 += 1
    l1 += 1
    
df1 = pd.DataFrame()
df1['level_1'] = list_level_1
column_names = ['level_2_' + str(level_2) for level_2 in list_level_2]
df2 = pd.DataFrame(data=mtx_sharpe,columns=column_names)
df_sharpe = pd.concat([df1,df2], axis=1) # axis=0 for rows, axis=1 for columns
df_sharpe = df_sharpe.dropna()
df_sharpe = df_sharpe.reset_index(drop=True)

df1 = pd.DataFrame()
df1['level_1'] = list_level_1
column_names = ['level_2_' + str(level_2) for level_2 in list_level_2]
df2 = pd.DataFrame(data=mtx_nb_trades,columns=column_names)
df_trades = pd.concat([df1,df2], axis=1) # axis=0 for rows, axis=1 for columns
df_trades = df_trades.dropna()
df_trades = df_trades.reset_index(drop=True)

print('Max Sharpe is ' + str(np.max(mtx_sharpe)))
