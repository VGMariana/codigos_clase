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

    

    