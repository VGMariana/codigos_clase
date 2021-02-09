# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 09:31:39 2020

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
scale = 252 # 1 252
notional = 10 # mnUSD
print('-----')
print('nb_decimals ' + str(nb_decimals))
print('scale ' + str(scale))

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
    
# # compute covariance matrix via np.cov
# returns = []
# for ric in rics:
#     x, x_str, t = stream_functions.load_timeseries(ric)
#     returns.append(x)
# mtx_covar = np.cov(returns) # cov = covariance 
# mtx_correl = np.corrcoef(returns) # corrcoef = correlation
# # this did not work, we need to synchronise the timeseries before computing the covar

# compute variance-covariance matrix by pairwise covariances
size = len(rics)
mtx_covar = np.zeros([size,size])
mtx_correl = np.zeros([size,size])
returns = []
for i in range(size):
    ric1 = rics[i]
    for j in range(i+1):
        ric2 = rics[j]
        ret1, ret2, t = stream_functions.synchronise_timeseries(ric1, ric2)
        returns = [ret1, ret2]
        # covariances
        temp_mtx = np.cov(returns)
        temp_covar = scale*temp_mtx[0][1]
        temp_covar = np.round(temp_covar,nb_decimals)
        mtx_covar[i][j] = temp_covar
        mtx_covar[j][i] = temp_covar
        # correlations
        temp_mtx = np.corrcoef(returns)
        temp_correl = temp_mtx[0][1]
        temp_correl = np.round(temp_correl,nb_decimals)
        mtx_correl[i][j] = temp_correl
        mtx_correl[j][i] = temp_correl

# # compute eigenvalues and eigenvectors
# eigenvalues, eigenvectors = LA.eig(mtx_covar)

# compute eigenvalues and eigenvectors for symmetric matrices
eigenvalues, eigenvectors = LA.eigh(mtx_covar)

# min-variance portfolio
print('----')
print('Min-variance portfolio:')
print('notional (mnUSD) = ' + str(notional))
variance_explained = eigenvalues[0] / sum(abs(eigenvalues))
eigenvector_min_variance = eigenvectors[:,0]
port_min_var = notional * eigenvector_min_variance / sum(abs(eigenvector_min_variance))
delta_min_var = sum(port_min_var)
print('delta (mnUSD) = ' + str(delta_min_var))
print('variance explained = ' + str(variance_explained))

# PCA (max-variance) portfolio
print('----')
print('PCA portfolio (max-variance):')
print('notional (mnUSD) = ' + str(notional))
variance_explained = eigenvalues[-1] / sum(abs(eigenvalues))
eigenvector_pca = eigenvectors[:,-1]
port_pca = notional * eigenvector_pca / sum(abs(eigenvector_pca))
delta_pca = sum(port_pca)
print('delta (mnUSD) = ' + str(delta_pca))
print('Variance explained by max eigenvector ' + str(variance_explained))

'''
Inestability of the variance-covariance matrix (with 6 banks):
nb_decimals = 3 and scale = 1 --> var-covar matrix is zero everywhere except one entry
nb_decimals = 4 and scale = 1 --> there is a negative eigenvalue
nb_decimals = 5 and scale = 1 --> eigenvectors of eig and eigh only differ by sign

scale = 252 and e1(n) the min eigenvalue with n decimals:
e1(4)/e1(3) has error 4.3%
e1(5)/e1(4) has error 0.28%
e1(6)/e1(5) has error 0.014%
'''



        
