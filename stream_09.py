# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 10:16:36 2020

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
nb_decimals = 6
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

# compute vectors of returns and volatilities for Markowitz portfolios
min_returns = np.min(port_mgr.returns)
max_returns = np.max(port_mgr.returns)
returns = min_returns + np.linspace(0.1,0.9,100) * (max_returns-min_returns)
volatilities = np.zeros([len(returns),1])
counter = 0
for target_return in returns:
    port_markowitz = port_mgr.compute_portfolio('markowitz', notional, target_return)
    volatilities[counter] = port_markowitz.volatility_annual
    counter += 1

# compute other portfolios
# black
label1 = 'markowitz-avg' # 'equi-weight'
port = port_mgr.compute_portfolio('markowitz', notional)
x1 = port.volatility_annual
y1 = port.return_annual
# red
label2 = 'long-only' # 'long-only' 'min-variance'
port = port_mgr.compute_portfolio(label2, notional)
x2 = port.volatility_annual
y2 = port.return_annual
# yellow
label3 = 'equi-weight' # 'equi-weight'
port = port_mgr.compute_portfolio(label3, notional)
x3 = port.volatility_annual
y3 = port.return_annual
# yellow
label4 = 'markowitz-target' # 'equi-weight'
port = port_mgr.compute_portfolio('markowitz', notional, target_return=0.23)
x4 = port.volatility_annual
y4 = port.return_annual
    
# plot Efficient Frontier
plt.figure()
plt.title('Efficient Frontier for a portfolio including ' + rics[0])
plt.scatter(volatilities,returns)
plt.plot(x1, y1, "ok", label=label1) # black
plt.plot(x2, y2, "or", label=label2) # red
plt.plot(x3, y3, "oy", label=label3) # yellow
plt.plot(x4, y4, "*k", label=label4) # yellow
plt.ylabel('portfolio return')
plt.xlabel('portfolio volatility')
plt.grid()
plt.legend(loc="best")
plt.show()


'''
References:

Modern Portfolio Theory:
https://en.wikipedia.org/wiki/Modern_portfolio_theory



'''