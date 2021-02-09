# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:37:27 2020

@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2

# import our own files and reload
import stream_functions
importlib.reload(stream_functions)
import stream_classes
importlib.reload(stream_classes)

# compute risk metrics for real returns
ric = 'SAN.MC' # DBK.DE ^IXIC MXN=X ^STOXX ^STOXX50E ^FCHI ^GDAXI ^S&P500 ^VIX
jb = stream_classes.jarque_bera_test(ric)
jb.load_timeseries()
jb.compute()
# jb.plot_timeseries()
jb.plot_histogram()
print(jb)
print('-----')

# # compute risk metrics for simulated returns
# ric = 'simulated'
# type_random_variable = 'normal' # normal exponential student chi-squared
# size = 10**6
# degrees_freedom = 9
# jb = stream_classes.jarque_bera_test(ric)
# jb.generate_random_vector(type_random_variable, size, degrees_freedom)
# jb.compute()
# jb.plot_histogram()
# print(jb)
# print('-----')
