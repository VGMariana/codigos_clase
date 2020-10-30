# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:39:01 2020

@author: Meva
"""

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
import stream_classes
importlib.reload(stream_classes)


def print_number(n=5):
    print(n)


def load_timeseries(ric, file_extension='csv'):
    # get market data
    # remember to modify the path to match your own directory
    path = 'C:\\Users\Meva\\.spyder-py3\\data\\' + ric + '.' + file_extension
    if file_extension == 'csv':
        table_raw = pd.read_csv(path) # default csv
    else:
        table_raw = pd.read_excel(path)
    # create table of returns
    t = pd.DataFrame()
    t['date'] = pd.to_datetime(table_raw['Date'], dayfirst=True)
    t['close'] = table_raw['Close']
    t.sort_values(by='date', ascending=True)
    t['close_previous'] = t['close'].shift(1)
    t['return_close'] = t['close']/t['close_previous'] - 1
    t = t.dropna()
    t = t.reset_index(drop=True)
    # input for Jarque-Bera test
    x = t['return_close'].values # returns as array
    x_str = 'Real returns ' + ric # label e.g. ric
    
    return x, x_str, t


def plot_timeseries_price(t, ric):
    # plot timeseries of price
    plt.figure()
    plt.plot(t['date'],t['close'])
    plt.title('Time series real prices ' + ric)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()
    
    
def plot_histogram(x, x_str, plot_str, bins=100):
    # plot histogram
    plt.figure()
    plt.hist(x,bins)
    plt.title('Histogram ' + x_str)
    plt.xlabel(plot_str)
    plt.show()
    

def synchronise_timeseries(ric, benchmark, file_extension='csv'):
    # loading data from csv or Excel file
    x1, str1, t1 = load_timeseries(ric, file_extension)
    x2, str2, t2 = load_timeseries(benchmark, file_extension)
    # synchronize timestamps
    timestamp1 = list(t1['date'].values)
    timestamp2 = list(t2['date'].values)
    timestamps = list(set(timestamp1) & set(timestamp2))
    # synchronised time series for x1 or ric
    t1_sync = t1[t1['date'].isin(timestamps)]
    t1_sync.sort_values(by='date', ascending=True)
    t1_sync = t1_sync.reset_index(drop=True)
    # synchronised time series for x2 or benchmark
    t2_sync = t2[t2['date'].isin(timestamps)]
    t2_sync.sort_values(by='date', ascending=True)
    t2_sync = t2_sync.reset_index(drop=True)
    # table of returns for ric and benchmark
    t = pd.DataFrame()
    t['date'] = t1_sync['date']
    t['price_1'] = t1_sync['close']
    t['price_2'] = t2_sync['close']
    t['return_1'] = t1_sync['return_close']
    t['return_2'] = t2_sync['return_close']
    # compute vectors of returns
    returns_ric = t['return_1'].values # variable y
    returns_benchmark = t['return_2'].values # variable x
    
    return returns_benchmark, returns_ric, t # x, y, t


def compute_beta(ric, benchmark, bool_print=False):
    capm = stream_classes.capm_manager(ric, benchmark)
    capm.load_timeseries()
    capm.compute()
    if bool_print:
        print('------')
        print(capm)
    beta = capm.beta
    return beta


def cost_function_beta_delta(x, delta, beta_usd, betas, epsilon=0.0):
    f_delta = (sum(x).item() + delta)**2
    f_beta = (np.transpose(betas).dot(x).item() + beta_usd)**2
    f_penalty = epsilon * sum(x**2).item()
    f = f_delta + f_beta + f_penalty
    return f