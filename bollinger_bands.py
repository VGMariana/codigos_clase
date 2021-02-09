# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 05:47:33 2020

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
import teacher_code
importlib.reload(teacher_code)
import student_code
importlib.reload(student_code)


class backtest:
    
    def __init__(self):
        self.ric_long = 'TOTF.PA' # numerator
        self.ric_short = 'REP.MC' # denominator
        self.rolling_days = 20 # N
        self.level_1 = 1. # a
        self.level_2 = 2. # b
        self.data_cut = 0.7 # 70% in-sample and 30% out-of-sample
        self.data_type = 'in-sample' # in-sample out-of-sample
        self.dataframe_data = pd.DataFrame() # dataframe of data
        self.dataframe_indicators = pd.DataFrame() # dataframe of indicators
        self.dataframe_strategy = pd.DataFrame() # dataframe of strategy
        self.pnl_mean = np.Infinity # annualised
        self.pnl_volatility = np.Infinity # annualised
        self.pnl_sharpe = np.Infinity # annualised
        self.nb_trades = 0
        
    
    def load_data(self):
        _, _, t = stream_functions.synchronise_timeseries(self.ric_long, self.ric_short)
        cut = int(self.data_cut*t.shape[0])
        if self.data_type == 'in-sample':
            df = t.head(cut)
        elif self.data_type == 'out-of-sample':
            df = t.tail(t.shape[0]-cut)
        df = df.reset_index(drop=True)
        # spread at current close
        df['spread'] = df['price_1']/df['price_2']
        base = df['spread'][0]
        df['spread'] = df['spread'] / base
        # spread at previous close
        df['spread_previous'] = df['price_1_previous']/df['price_2_previous']
        df['spread_previous'] = df['spread_previous'] / base
        self.dataframe_data = df
        
        
    def compute_indicators(self, bool_plot=False):
        size = self.dataframe_data.shape[0]
        columns = ['lower_2','lower_1','mean','upper_1','upper_2']
        mtx_bollinger = np.empty((size,len(columns)))
        mtx_bollinger[:] = np.nan
        for n in range(self.rolling_days-1,size):
            vec_price = self.dataframe_data['spread'].values
            vec_price = vec_price[n-self.rolling_days+1:n+1]
            mu = np.mean(vec_price)
            sigma = np.std(vec_price)
            m = 0
            mtx_bollinger[n][m] = mu - self.level_2*sigma
            m = m + 1
            mtx_bollinger[n][m] = mu - self.level_1*sigma
            m = m + 1
            mtx_bollinger[n][m] = mu
            m = m + 1
            mtx_bollinger[n][m] = mu + self.level_1*sigma
            m = m + 1
            mtx_bollinger[n][m] = mu + self.level_2*sigma
            m = m + 1
        df1 = pd.DataFrame(data=mtx_bollinger,columns=columns)
        df2 = pd.concat([self.dataframe_data,df1], axis=1) # axis=0 for rows, axis=1 for columns
        df2 = df2.dropna()
        df2 = df2.reset_index(drop=True)
        self.dataframe_indicators = df2
        if bool_plot:
            self.plot_indicators()
            
    
    def run_strategy(self, bool_plot=False):
        # loop for backtest
        size = self.dataframe_indicators.shape[0]
        columns = ['position','entry_signal','exit_signal','pnl_daily','trade','pnl_trade']
        position = 0
        entry_spread = 0.
        can_trade = False
        size = self.dataframe_indicators.shape[0]
        mtx_backtest = np.zeros((size,len(columns)))
        for n in range(size):
            # input data for the day
            spread = self.dataframe_indicators['spread'][n]
            spread_previous = self.dataframe_indicators['spread_previous'][n]
            lower_2 = self.dataframe_indicators['lower_2'][n]
            lower_1 = self.dataframe_indicators['lower_1'][n]
            mean = self.dataframe_indicators['mean'][n]
            upper_1 = self.dataframe_indicators['upper_1'][n]
            upper_2 = self.dataframe_indicators['upper_2'][n]
            # reset output data for the day
            pnl_daily = 0.
            trade = 0
            pnl_trade = 0.
            # check if we can trade
            if not can_trade:
                can_trade = position == 0 and spread > lower_1 and spread < upper_1
            if not can_trade:
                continue
            # enter new position
            if position == 0: 
                entry_signal = 0
                exit_signal = 0
                if spread > lower_2 and spread < lower_1:
                    entry_signal = 1 # buy signal
                    position = 1
                    entry_spread = spread
                elif spread > upper_1 and spread < upper_2:
                    entry_signal = -1 # sell signal
                    position = -1
                    entry_spread = spread
            # exit long position
            elif position == 1:
                entry_signal = 0
                pnl_daily = position*(spread - spread_previous)
                if n == size-1 or spread > mean or spread < lower_2:
                    exit_signal = 1 # last day or take profit or stop loss
                    pnl_trade = position*(spread - entry_spread)
                    position = 0
                    trade = 1
                    can_trade = False
                else:
                    exit_signal = 0
            # exit short position
            elif position == -1:
                entry_signal = 0
                pnl_daily = position*(spread - spread_previous)
                if n == size-1 or spread < mean or spread > upper_2:
                    exit_signal = 1 # last day or take profit or stop loss
                    pnl_trade = position*(spread - entry_spread)
                    position = 0
                    trade = 1
                    can_trade = False
                else:
                    exit_signal = 0
            # save data for the day
            m = 0
            mtx_backtest[n][m] = position
            m = m + 1
            mtx_backtest[n][m] = entry_signal
            m = m + 1
            mtx_backtest[n][m] = exit_signal
            m = m + 1
            mtx_backtest[n][m] = pnl_daily
            m = m + 1
            mtx_backtest[n][m] = trade
            m = m + 1
            mtx_backtest[n][m] = pnl_trade
        df1 = pd.DataFrame(data=mtx_backtest,columns=columns)
        df2 = pd.concat([self.dataframe_indicators,df1], axis=1) # axis=0 for rows, axis=1 for columns
        df2 = df2.dropna()
        df2 = df2.reset_index(drop=True)
        df2['cum_pnl_daily'] = np.cumsum(df2['pnl_daily'])
        self.dataframe_strategy = df2
        # compute Sharpe ratio and number of trades
        vec_pnl = df2['pnl_daily'].values
        self.pnl_mean = np.round(np.mean(vec_pnl) * 252, 4)
        self.pnl_volatility = np.round(np.std(vec_pnl) * np.sqrt(252), 4)
        self.pnl_sharpe = np.round(self.pnl_mean / self.pnl_volatility, 4)
        df3 = df2[df2['trade'] == 1]
        self.nb_trades = df3.shape[0]
        if bool_plot:
            self.plot_strategy()
        
        
    def plot_indicators(self):
        timestamps = self.dataframe_indicators['date']
        spread = self.dataframe_indicators['spread']
        mu = self.dataframe_indicators['mean']
        u1 = self.dataframe_indicators['upper_1']
        u2 = self.dataframe_indicators['upper_2']
        l1 = self.dataframe_indicators['lower_1']
        l2 = self.dataframe_indicators['lower_2']
        plt.figure()
        plt.title('Spread ' + self.ric_long + ' / ' + self.ric_short)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.plot(timestamps, mu, color='blue', label='mean')
        plt.plot(timestamps, l1, color='green', label='lower_1')
        plt.plot(timestamps, u1, color='green', label='upper_1')
        plt.plot(timestamps, l2, color='red', label='lower_2')
        plt.plot(timestamps, u2, color='red', label='upper_2')
        plt.plot(timestamps, spread, color='black', label='spread')
        plt.legend(loc=0)
        plt.grid()
        plt.show()
        
        
    def plot_strategy(self):
        plot_str = 'Cumulative PNL ' + str(self.ric_long) + ' / ' + str(self.ric_short) + '\n'\
            + 'rolling days ' + str(self.rolling_days) + ' | level_1 ' + str(self.level_1) + ' | level_2 ' + str(self.level_2) + '\n'\
            + 'pnl annual: mean ' + str(self.pnl_mean) + ' | volatility ' + str(self.pnl_volatility) + ' | Sharpe ' + str(self.pnl_sharpe)
        plt.figure()
        plt.title(plot_str)
        plt.xlabel('Time')
        plt.ylabel('cum PNL')
        plt.plot(self.dataframe_strategy['date'], self.dataframe_strategy['cum_pnl_daily'], color='blue', label='mean')
        plt.grid()
        plt.show()

    

    