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
    
    def __init__(self, benchmark, ric):
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
        self.correlation = None
        self.predictor_linreg = [] # y = alpha + beta*x
        
        
    def __str__(self):
        # str_self = '__str__ not yet defined, next course please'
        str_self = 'Linear regression | ric ' + self.ric\
            + ' | benchmark ' + self.benchmark + '\n'\
            + 'alpha (intercept) ' + str(self.alpha)\
            + ' | beta (slope) ' + str(self.beta) + '\n'\
            + 'p-value ' + str(self.p_value)\
            + ' | null hypothesis ' + str(self.null_hypothesis) + '\n'\
            + 'r-value (correlation) ' + str(self.r_value)\
            + ' | r-squared ' + str(self.r_squared)
        return str_self
        
    
    def load_timeseries(self):
        self.returns_benchmark, self.returns_ric, self.dataframe\
            = stream_functions.synchronise_timeseries(self.benchmark, self.ric)
    
    
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
        self.correlation = self.r_value
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
        price_benchmark = self.dataframe['price_1']
        price_ric = self.dataframe['price_2'] 
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
                                  color='blue', label=self.benchmark)
        ax2 = self.dataframe.plot(kind='line', x='date', y='price_2', ax=ax, grid=True,\
                                  color='red', secondary_y=True, label=self.ric)
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.show()
        
        
class hedge_manager():
    
    def __init__(self, benchmark, ric, hedge_rics, delta):
        self.benchmark = benchmark
        self.ric = ric
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
        self.beta = stream_functions.compute_beta(self.benchmark, self.ric)
        self.beta_usd = self.beta*self.delta
        betas = [stream_functions.compute_beta(self.benchmark, hedge_ric) \
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
            
            
class portfolio_manager:
    
    def __init__(self, rics, nb_decimals):
        self.rics = rics
        self.nb_decimals = nb_decimals
        self.covariance_matrix = [] # annualised
        self.correlation_matrix = [] # annualised
        self.returns = [] # annualised
        self.volatilities = [] # annualised
        
        
    def compute_covariance_matrix(self, bool_print=False):
        # compute variance-covariance matrix by pairwise covariances
        scale = 252 # annualised
        size = len(self.rics)
        mtx_covar = np.zeros([size,size])
        mtx_correl = np.zeros([size,size])
        vec_returns = np.zeros([size,1])
        vec_volatilities = np.zeros([size,1])
        returns = []
        for i in range(size):
            ric1 = self.rics[i]
            temp_ret = []
            for j in range(i+1):
                ric2 = self.rics[j]
                ret1, ret2, t = stream_functions.synchronise_timeseries(ric1, ric2)
                returns = [ret1, ret2]
                # covariances
                temp_mtx = np.cov(returns)
                temp_covar = scale*temp_mtx[0][1]
                temp_covar = np.round(temp_covar,self.nb_decimals)
                mtx_covar[i][j] = temp_covar
                mtx_covar[j][i] = temp_covar
                # correlations
                temp_mtx = np.corrcoef(returns)
                temp_correl = temp_mtx[0][1]
                temp_correl = np.round(temp_correl,self.nb_decimals)
                mtx_correl[i][j] = temp_correl
                mtx_correl[j][i] = temp_correl
                if j == 0:
                    temp_ret = ret1
            # returns
            temp_mean = np.round(scale*np.mean(temp_ret), self.nb_decimals)
            vec_returns[i] = temp_mean
            # volatilities
            temp_volatility = np.round(np.sqrt(scale)*np.std(temp_ret), self.nb_decimals)
            vec_volatilities[i] = temp_volatility
            
        self.covariance_matrix = mtx_covar
        self.correlation_matrix = mtx_correl
        self.returns = vec_returns
        self.volatilities = vec_volatilities
        
        if bool_print:
            print('-----')
            print('Portfolio Manager details:')
            print('Securities:')
            print(self.rics)
            print('Returns (annualised):')
            print(self.returns)
            print('Volatilities (annualised):')
            print(self.volatilities)
            print('Variance-covariance matrix (annualised):')
            print(self.covariance_matrix)
            print('Correlation matrix:')
            print(self.correlation_matrix)
            
            
    def compute_portfolio(self, portfolio_type, notional, target_return=None):
        
        size = len(self.rics)
        port_item = portfolio_item(self.rics, notional)
        
        if portfolio_type == 'min-variance':
            port_min_variance, variance_explained = \
                stream_functions.compute_portfolio_min_variance(self.covariance_matrix, notional)
            port_item.type = portfolio_type
            port_item.weights = port_min_variance
            port_item.variance_explained = variance_explained
            
        elif portfolio_type == 'pca' or portfolio_type == 'max-variance':
            port_pca, variance_explained = \
                stream_functions.compute_portfolio_pca(self.covariance_matrix, notional)
            port_item.type = 'pca'
            port_item.weights = port_pca
            port_item.variance_explained = variance_explained
            
        elif portfolio_type == 'long-only':
            port_long_only = stream_functions.compute_portfolio_long_only(size, notional, self.covariance_matrix)
            port_item.type = 'long-only'
            port_item.weights = port_long_only
            
        elif portfolio_type == 'markowitz':
            if target_return == None:
                target_return = np.mean(self.returns) #annualised
            port_markowitz = stream_functions.compute_portfolio_markowitz(size, notional, self.covariance_matrix,\
                                                                   self.returns, target_return)
            port_item.type = 'markowitz | target return ' + str(target_return) 
            port_item.weights = port_markowitz
            port_item.target_return = target_return
            
        else:
            size = len(self.rics)
            port_equi = stream_functions.compute_portfolio_equi_weight(size, notional)
            port_item.type = 'equi-weight'
            port_item.weights = port_equi
        
        port_item.delta = sum(port_item.weights)
        port_item.pnl_annual = np.dot(port_item.weights.T,self.returns).item()
        port_item.return_annual = port_item.pnl_annual / notional
        port_item.volatility_annual = \
            stream_functions.compute_portfolio_volatility(self.covariance_matrix, port_item.weights)
        if port_item.volatility_annual > 0.0:
            port_item.sharpe_annual =  port_item.return_annual / port_item.volatility_annual
            
        return port_item
            
            
class portfolio_item():
    
    def __init__(self, rics, notional):
        self.rics = rics
        self.notional = notional
        self.type = ''
        self.weights = []
        self.delta = 0.0
        self.pnl_annual = 0.0
        self.return_annual = 0.0
        self.volatility_annual = 0.0
        self.sharpe_annual = 0.0
        self.variance_explained = None
        self.target_return = None


    def summary(self):
        print('-----')
        print('Portfolio type: ' + self.type)
        print('Rics:')
        print(self.rics)
        print('Weights:')
        print(self.weights)
        if not self.variance_explained == None:
            print('Variance explained: ' + str(self.variance_explained))
        print('Notional (mnUSD): ' + str(self.notional))
        print('Delta (mnUSD): ' + str(self.delta))
        print('PnL annual (mnUSD): ' + str(self.pnl_annual))
        print('Return annual (mnUSD): ' + str(self.return_annual))
        if not self.target_return == None:
            print('Target Return: ' + str(self.target_return))
        print('Volatility annual (mnUSD): ' + str(self.volatility_annual))
        print('Sharpe ratio annual: ' + str(self.sharpe_annual))


        