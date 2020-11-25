# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 18:15:41 2020

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

# input parameters
nb_decimals = 6 # 3 4 5 6
notional = 10 # mnUSD
print('-----')
print('inputs:')
print('nb_decimals ' + str(nb_decimals))
print('notional ' + str(notional))

# rics = ['SAN.MC',\
#         'BBVA.MC',\
#         'SOGN.PA',\
#         'BNPP.PA',\
#         'INGA.AS',\
#         'KBC.BR']
# rics = ['MXNUSD=X',\
#         'EURUSD=X',\
#         'GBPUSD=X',\
#         'CHFUSD=X']
# rics = ['SAN.MC',\
#         'BBVA.MC',\
#         'SOGN.PA',\
#         'BNPP.PA',\
#         'INGA.AS',\
#         'KBC.BR',\
#         'CRDI.MI',\
#         'ISP.MI',\
#         'DBKGn.DE',\
#         'CBKG.DE']
# rics = ['SAN.MC',\
#         'BBVA.MC',\
#         'SOGN.PA',\
#         'BNPP.PA']
# rics = ['^S&P500',\
#         '^VIX']
# rics = ['SAN.MC',\
#         'BBVA.MC']
# rics = ['SGREN.MC',\
#         'VWS.CO',\
#         'TOTF.PA',\
#         'REP.MC',\
#         'BP.L',\
#         'RDSa.AS',\
#         'RDSa.L']
rics = ['AAL.L',\
        'ANTO.L',\
        'GLEN.L',\
        'MT.AS',\
        'RIO.L']

# compute covariance matrix
port_mgr = stream_classes.portfolio_manager(rics, nb_decimals)
port_mgr.compute_covariance_matrix(bool_print=True)

# compute min-variance portfolio
portfolio_min_variance = port_mgr.compute_portfolio('min-variance', notional)
portfolio_min_variance.summary()

# compute PCA or max-variance portfolio
portfolio_pca = port_mgr.compute_portfolio('pca', notional)
portfolio_pca.summary()

# compute default portfolio, which is equi-weight
portfolio_equiweight = port_mgr.compute_portfolio('default', notional)
portfolio_equiweight.summary()

# compute long-only portfolio with minimal variance
portfolio_equiweight = port_mgr.compute_portfolio('long-only', notional)
portfolio_equiweight.summary()

# compute long-only Markowitz portfolio
portfolio_markowitz = port_mgr.compute_portfolio('markowitz', notional, target_return=0.20)
portfolio_markowitz.summary()



