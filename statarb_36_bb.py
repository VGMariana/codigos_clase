# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 09:59:51 2021

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
universe = ['SAN.MC',\
        'BBVA.MC',\
        'SOGN.PA',\
        'BNPP.PA',\
        'INGA.AS',\
        'KBC.BR',\
        'CRDI.MI',\
        'ISP.MI',\
        'DBKGn.DE',\
        'CBKG.DE',\
        'AAL.L',\
        'GLEN.L',\
        'RIO.L',\
        'MT.AS',\
        'SGREN.MC',\
        'VWS.CO',\
        'TOTF.PA',\
        'REP.MC',\
        'BP.L',\
        'RDSa.AS',\
        'RDSa.L',\
        'EQNR.OL',\
        'EONGn.DE',\
        'RWEG.DE',\
        'EDP.LS',\
        'EDPR.LS',\
        'EDF.PA',\
        '^FCHI',\
        '^GDAXI',\
        '^S&P500',\
        '^NASDAQ',\
        '^STOXX',\
        '^STOXX50E',\
        '^VIX',\
        'MXNUSD=X',\
        'EURUSD=X',\
        'GBPUSD=X',\
        'CHFUSD=X']
backtest = bollinger_bands.backtest()
backtest.data_cut = 0.7
backtest.data_type = 'in-sample' # in-sample out-of-sample

# list_rolling_days = [15,16,17,18,19,20,21,22,23,24,25]
# list_level_1 = [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4]

list_rolling_days = [int(x) for x in np.linspace(10, 60, num=11)]
list_level_1 = [np.round(x,2) for x in np.linspace(0.7, 1.5, num=9)]

mtx_sharpe = np.zeros((len(list_rolling_days),len(list_level_1)))
mtx_nb_trades = np.zeros((len(list_rolling_days),len(list_level_1)))

for ric1 in universe:
    for ric2 in universe:
        backtest.ric_long = ric1
        backtest.ric_short = ric2       
        rd = 0
        for rolling_days in list_rolling_days:
            l1 = 0
            for level_1 in list_level_1:
                # parameters to optimise
                backtest.rolling_days = rolling_days
                backtest.level_1 = level_1
                backtest.level_2 = 2*level_1
                # load data
                backtest.load_data()
                # compute indicator
                backtest.compute_indicators()
                # run backtest of trading strategy
                backtest.run_strategy()
                # get results in a dataframe format
                df_strategy = backtest.dataframe_strategy
                # save data in tables
                mtx_sharpe[rd][l1] = backtest.pnl_sharpe
                mtx_nb_trades[rd][l1] = backtest.nb_trades
                l1 += 1
            rd += 1
            
        df1 = pd.DataFrame()
        df1['rolling_days'] = list_rolling_days
        column_names = ['level_1_' + str(level_1) for level_1 in list_level_1]
        df2 = pd.DataFrame(data=mtx_sharpe,columns=column_names)
        df_sharpe = pd.concat([df1,df2], axis=1) # axis=0 for rows, axis=1 for columns
        df_sharpe = df_sharpe.dropna()
        df_sharpe = df_sharpe.reset_index(drop=True)
        
        df1 = pd.DataFrame()
        df1['rolling_days'] = list_rolling_days
        column_names = ['level_1_' + str(level_1) for level_1 in list_level_1]
        df2 = pd.DataFrame(data=mtx_nb_trades,columns=column_names)
        df_trades = pd.concat([df1,df2], axis=1) # axis=0 for rows, axis=1 for columns
        df_trades = df_trades.dropna()
        df_trades = df_trades.reset_index(drop=True)
        
        print('Max Sharpe is ' + str(np.max(mtx_sharpe)))
        
        # this is pseudo code: if you run it it will not work
        # you need to create a table with 3 columns: ric1, ric2 and sharpe
        # we run this during the weekend and find potential good pairs to zoom in
        super_sharpes['ric1']['ric2'] = np.max(mtx_sharpe)

