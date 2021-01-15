# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 09:14:21 2021

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
backtest.rolling_days = 35
backtest.level_1 = 0.65
backtest.level_2 = 1.3
backtest.data_cut = 0.7
backtest.data_type = 'out-of-sample' # in-sample out-of-sample


# load data
backtest.load_data()

# compute indicator
backtest.compute_indicators(bool_plot=True)

# run backtest of trading strategy
backtest.run_strategy(bool_plot=True)

# get results in a dataframe format
df_strategy = backtest.dataframe_strategy
