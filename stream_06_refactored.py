# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 18:56:58 2020

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

# import our own files and reload
import stream_functions
importlib.reload(stream_functions)
import stream_classes
importlib.reload(stream_classes)

# input parameters
ric = 'BBVA.MC'
benchmark = '^STOXX50E'
# hedge_rics =  ['SAN.MC','REP.MC']
# hedge_rics = ['BBVA.MC','REP.MC']
# hedge_rics = ['SAN.MC','DBKGn.DE']
# hedge_rics = ['^FCHI','^GDAXI']
# hedge_rics = ['^STOXX','SAN.MC']
# hedge_rics = ['^S&P500','^NASDAQ']
# hedge_rics = ['^STOXX50E','^NASDAQ']
hedge_rics = ['SAN.MC','^FCHI','^GDAXI']
# hedge_rics = ['SAN.MC']
delta = 10

# compute optimal hedge
hedger = stream_classes.hedge_manager(benchmark, ric, hedge_rics, delta)
hedger.load_inputs(bool_print=True)
# hedger.compute_exact(bool_print=True)
hedger.compute_numerical(epsilon=0.01, bool_print=True)
# optimal_hedge = hedger.dataframe

      
