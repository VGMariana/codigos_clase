# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 09:10:59 2020

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

# import our own files and reload
import stream_functions
importlib.reload(stream_functions)


class jarque_bera_test():
    
    def __init__(self, ric):
        self.ric = ric
        self.returns = []
        self.dataframe = pd.DataFrame()
        self.size = 0
        self.str_name = ''
        self.mean = 0.0
        self.std = 0.0
        self.skew = 0.0
        self.kurt = 0.0
        self.median = 0.0
        self.var_95 = 0.0
        self.cvar_95 = 0.0
        self.sharpe = 0.0
        self.jarque_bera = 0.0
        self.p_value = 0.0
        self.is_normal = 0.0
        
        
    def __str__(self):
        str_self = self.str_name + ' | size ' + str(self.size) + '\n' + self.plot_str()
        return str_self


    def load_timeseries(self):
        self.returns, self.str_name, self.dataframe = stream_functions.load_timeseries(self.ric)
        self.size = len(self.returns)
        
        
    def generate_random_vector(self, type_random_variable, size=10**6, degrees_freedom=None):
        self.size = size
        if type_random_variable == 'normal':
            self.returns = np.random.standard_normal(size)
            self.str_name = 'Standard Normal RV'
        elif type_random_variable == 'exponential':
            self.returns = np.random.standard_exponential(size)
            self.str_name = 'Exponential RV'
        elif type_random_variable == 'student':
            if degrees_freedom == None:
                degrees_freedom = 750 # borderline for Jarque-Bera with 10**6 samples
            self.returns = np.random.standard_t(df=degrees_freedom,size=size)
            self.str_name = 'Student RV (df = ' + str(degrees_freedom) + ')'
        elif type_random_variable == 'chi-squared':
            if degrees_freedom == None:
                degrees_freedom = 2 # Jarque-Bera test uses 2 degrees of freedom
            self.returns = np.random.chisquare(df=degrees_freedom,size=size)
            self.str_name = 'Chi-squared RV (df = ' + str(degrees_freedom) + ')'


    def compute(self):
        self.mean = np.mean(self.returns)
        self.std = np.std(self.returns) # volatility
        self.skew = skew(self.returns)
        self.kurt = kurtosis(self.returns) # excess kurtosis
        self.sharpe = self.mean / self.std * np.sqrt(252) # annualised
        self.median = np.median(self.returns)
        self.var_95 = np.percentile(self.returns,5)
        self.cvar_95 = np.mean(self.returns[self.returns <= self.var_95])
        self.jarque_bera = self.size/6*(self.skew**2 + 1/4*self.kurt**2)
        self.p_value = 1 - chi2.cdf(self.jarque_bera, df=2)
        self.is_normal = (self.p_value > 0.05) # equivalently jb < 6


    def plot_str(self):
        nb_decimals = 4
        plot_str = 'mean ' + str(np.round(self.mean,nb_decimals))\
            + ' | median ' + str(np.round(self.median,nb_decimals))\
            + ' | std dev ' + str(np.round(self.std,nb_decimals))\
            + ' | skewness ' + str(np.round(self.skew,nb_decimals)) + '\n'\
            + 'kurtosis ' + str(np.round(self.kurt,nb_decimals))\
            + ' | Sharpe ratio ' + str(np.round(self.sharpe,nb_decimals))\
            + ' | VaR 95% ' + str(np.round(self.var_95,nb_decimals))\
            + ' | CVaR 95% ' + str(np.round(self.cvar_95,nb_decimals)) + '\n'\
            + 'Jarque Bera ' + str(np.round(self.jarque_bera,nb_decimals))\
            + ' | p-value ' + str(np.round(self.p_value,nb_decimals))\
            + ' | is normal ' + str(self.is_normal)
        return plot_str
    
    
    def plot_timeseries(self):
        stream_functions.plot_timeseries_price(self.dataframe, self.ric)
        
    
    def plot_histogram(self):
        stream_functions.plot_histogram(self.returns, self.str_name, self.plot_str())
    
    
class capm_manager():
    
    def __init__(self, ric, benchmark):
        self.ric = ric
        self.benchmark = benchmark
        self.returns_benchmark = [] # x
        self.returns_ric = [] # y
        self.dataframe = pd.DataFrame()
        self.alpha = None
        self.beta = None
        self.p_value = None
        self.null_hypothesis = False
        self.r_value = None
        self.r_squared = None
        self.predictor_linreg = [] # y = alpha + beta*x
        
        
    def __str__(self):
        # str_self = '__str__ not yet defined, next course please'
        str_self = 'Linear regression | ric ' + self.ric\
            + ' | benchmark ' + self.benchmark + '\n'\
            + 'alpha (intercept) ' + str(self.alpha)\
            + ' | beta (slope) ' + str(self.beta) + '\n'\
            + 'p-value ' + str(self.p_value)\
            + ' | null hypothesis ' + str(self.null_hypothesis) + '\n'\
            + 'r-value ' + str(self.r_value)\
            + ' | r-squared ' + str(self.r_squared)
        return str_self
        
    
    def load_timeseries(self):
        self.returns_benchmark, self.returns_ric, self.dataframe\
            = stream_functions.synchronise_timeseries(self.ric, self.benchmark)
    
    
    def compute(self):
        # linear regression of ric with respect to benchmark
        nb_decimals = 4
        slope, intercept, r_value, p_value, std_err\
            = linregress(self.returns_benchmark,self.returns_ric)
        self.beta = np.round(slope, nb_decimals)
        self.alpha = np.round(intercept, nb_decimals)
        self.p_value = np.round(p_value, nb_decimals) 
        self.null_hypothesis = p_value > 0.05 # p_value < 0.05 --> reject null hypothesis
        self.r_value = np.round(r_value, nb_decimals) # correlation coefficient
        self.r_squared = np.round(r_value**2, nb_decimals) # pct of variance of y explained by x
        self.predictor_linreg = self.alpha + self.beta*self.returns_benchmark
        
        
    def scatterplot(self):
        # scatterplot of returns
        str_title = 'Scatterplot of returns' + '\n' + self.__str__()
        plt.figure()
        plt.title(str_title)
        plt.scatter(self.returns_benchmark,self.returns_ric)
        plt.plot(self.returns_benchmark, self.predictor_linreg, color='green')
        plt.ylabel(self.ric)
        plt.xlabel(self.benchmark)
        plt.grid()
        plt.show()
        
        
    def plot_normalised(self):
        # plot 2 timeseries normalised at 100
        timestamps = self.dataframe['date']
        price_ric = self.dataframe['price_1']
        price_benchmark = self.dataframe['price_2'] 
        plt.figure(figsize=(12,5))
        plt.title('Time series of prices | normalised at 100')
        plt.xlabel('Time')
        plt.ylabel('Normalised prices')
        price_ric = 100 * price_ric / price_ric[0]
        price_benchmark = 100 * price_benchmark / price_benchmark[0]
        plt.plot(timestamps, price_ric, color='blue', label=self.ric)
        plt.plot(timestamps, price_benchmark, color='red', label=self.benchmark)
        plt.legend(loc=0)
        plt.grid()
        plt.show()
        
        
    def plot_dual_axes(self):
        # plot 2 timeseries with 2 vertical axes
        plt.figure(figsize=(12,5))
        plt.title('Time series of prices')
        plt.xlabel('Time')
        plt.ylabel('Prices')
        ax = plt.gca()
        ax1 = self.dataframe.plot(kind='line', x='date', y='price_1', ax=ax, grid=True,\
                                  color='blue', label=self.ric)
        ax2 = self.dataframe.plot(kind='line', x='date', y='price_2', ax=ax, grid=True,\
                                  color='red', secondary_y=True, label=self.benchmark)
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.show()
        
        
class hedge_manager():
    
    def __init__(self, ric, benchmark, hedge_rics, delta):
        self.ric = ric
        self.benchmark = benchmark
        self.hedge_rics = hedge_rics
        self.delta = delta
        self.beta = None
        self.beta_usd = None
        self.dataframe = pd.DataFrame()
        self.hedge_delta = None
        self.hedge_beta_usd = None
        self.betas = []
        self.optimal_hedge = []
        
        
    def load_inputs(self, bool_print=False):
        self.beta = stream_functions.compute_beta(self.ric, self.benchmark)
        self.beta_usd = self.beta*self.delta
        betas = [stream_functions.compute_beta(hedge_ric, self.benchmark) \
                       for hedge_ric in self.hedge_rics]
        self.betas = np.asarray(betas).reshape([len(self.hedge_rics),1])
        self.dataframe['ric'] = self.hedge_rics
        self.dataframe['beta'] = self.betas
        if bool_print:
            print('------')
            print('Input portfolio:')
            print('Delta mnUSD for ' + self.ric + ' is ' + str(self.delta))
            print('Beta for ' + self.ric + ' vs ' + self.benchmark + ' is ' + str(self.beta))
            print('Beta mnUSD for ' + self.ric + ' vs ' + self.benchmark + ' is ' + str(self.beta_usd))
            print('------')
            print('Input hedges:')
            for n in range(self.dataframe.shape[0]):
                print('Beta for hedge[' + str(n) + '] = ' + self.dataframe['ric'][n] \
                      + ' vs ' + self.benchmark + ' is ' + str(self.dataframe['beta'][n]))

                    
    def compute_exact(self, bool_print=False):
        size = len(self.hedge_rics)
        if not size == 2:
            print('------')
            print('Warning: cannot compute exact solution, hedge_rics size ' + str(size) + ' =/= 2')
            return
        deltas = np.ones([size,1])
        targets = -np.array([[self.delta],[self.beta_usd]])
        mtx = np.transpose(np.column_stack((deltas,self.betas)))
        self.optimal_hedge = np.linalg.inv(mtx).dot(targets)
        self.dataframe['delta'] = self.optimal_hedge
        self.dataframe['beta_usd'] = self.betas*self.optimal_hedge
        self.hedge_delta = np.sum(self.dataframe['delta'])
        self.hedge_beta_usd = np.sum(self.dataframe['beta_usd'])
        if bool_print:
            self.print_output('Exact solution from linear algebra')
            

    def print_output(self, optimisation_type):
            print('------')
            print('Optimisation result | ' + optimisation_type + ':')
            print('------')
            print('Delta: ' + str(self.delta))
            print('Beta USD: ' + str(self.beta_usd))
            print('------')
            print('Hedge delta: ' + str(self.hedge_delta))
            print('Hedge beta USD: ' + str(self.hedge_beta_usd))
            print('------')
            print('Betas for the hedge:')
            print(self.betas)
            print('------')
            print('Optimal hedge:')
            print(self.optimal_hedge)
            

    def compute_numerical(self, epsilon=0.0, bool_print=False):
        x = np.zeros([len(self.betas),1])
        args = (self.delta, self.beta_usd, self.betas, epsilon)
        optimal_result = minimize(fun=stream_functions.cost_function_beta_delta,\
                                  x0=x, args=args, method='BFGS')
        self.optimal_hedge = optimal_result.x.reshape([len(self.betas),1])
        self.dataframe['delta'] = self.optimal_hedge
        self.dataframe['beta_usd'] = self.betas*self.optimal_hedge
        self.hedge_delta = np.sum(self.dataframe['delta'])
        self.hedge_beta_usd = np.sum(self.dataframe['beta_usd'])
        if bool_print:
            self.print_output('Numerical solution with optimize.minimize')