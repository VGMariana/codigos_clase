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
    

def synchronise_timeseries(benchmark, ric, file_extension='csv'):
    # loading data from csv or Excel file
    x1, str1, t1 = load_timeseries(benchmark, file_extension)
    x2, str2, t2 = load_timeseries(ric, file_extension)
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
    t['price_1'] = t1_sync['close'] # price benchmark
    t['price_2'] = t2_sync['close'] # price ric
    t['return_1'] = t1_sync['return_close'] # return benchmark
    t['return_2'] = t2_sync['return_close'] #return ric
    # compute vectors of returns
    returns_benchmark = t['return_1'].values # variable x
    returns_ric = t['return_2'].values # variable y
    return returns_benchmark, returns_ric, t # x, y, t


def compute_beta(benchmark, ric, bool_print=False):
    capm = stream_classes.capm_manager(benchmark, ric)
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


def compute_portfolio_min_variance(covariance_matrix, notional):
    eigenvalues, eigenvectors = LA.eigh(covariance_matrix)
    variance_explained = eigenvalues[0] / sum(abs(eigenvalues))
    eigenvector = eigenvectors[:,0]
    if max(eigenvector) < 0.0:
        eigenvector = - eigenvector
    port_min_variance = notional * eigenvector / sum(abs(eigenvector))
    return port_min_variance, variance_explained


def compute_portfolio_pca(covariance_matrix, notional):
    eigenvalues, eigenvectors = LA.eigh(covariance_matrix)
    variance_explained = eigenvalues[-1] / sum(abs(eigenvalues))
    eigenvector = eigenvectors[:,-1]
    if max(eigenvector) < 0.0:
        eigenvector = - eigenvector
    port_pca = notional * eigenvector / sum(abs(eigenvector))
    return port_pca, variance_explained


def compute_portfolio_equi_weight(size, notional):
    port_equi = (notional / size) * np.ones([size])
    return port_equi


def compute_portfolio_long_only(size, notional, covariance_matrix):
    # initialise optimisation
    x = np.zeros([size,1])
    # initialise constraints
    cons = [{"type": "eq", "fun": lambda x: sum(abs(x)) - 1}]
    bnds = [(0, None) for i in range(size)]
    # compute optimisation
    res = minimize(compute_portfolio_variance, x, args=(covariance_matrix), constraints=cons, bounds=bnds)
    port_long_only = notional * res.x
    return port_long_only


def compute_portfolio_variance(x, covariance_matrix):
    variance = np.dot(x.T, np.dot(covariance_matrix, x)).item()
    return variance


def compute_portfolio_volatility(covariance_matrix, weights):
    notional = sum(abs(weights))
    if notional <= 0.0:
        return 0.0
    weights = weights / notional # unitary weights in the L1-norm
    variance = np.dot(weights.T, np.dot(covariance_matrix, weights)).item()
    if variance <= 0.0:
        return 0.0
    volatility = np.sqrt(variance)
    return volatility