# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 18:55:46 2020

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


# define the function to minimise
def cost_function(x, roots, coeffs):
    f = 0
    for n in range(len(x)):
        f += coeffs[n]*(x[n] - roots[n])**2
    return f

# input parameters
roots = np.random.randint(low=-20, high=20, size=5)
# roots =  np.array([2.,-1.])
coeffs = np.ones([len(roots),1])

# initialise optimisation
x = np.zeros([len(roots),1])
method = 'BFGS' # BFGS 

# compute optimisation
optimal_result = minimize(fun=cost_function, x0=x, args=(roots,coeffs), method=method)

# print
print('------')
print('Optimisation result:')
print(optimal_result)
print('------')
print('Roots:')
print(roots)
print('------')

'''
Comments on the method for minimize:
The BFGS method does not require to provide explicitly the Jacobian or Hessian.
There are other methods, but I use this one for my optimisers in trading.
It computes numerically the Jacobian and Hessian: ideal for quadratic optimisations
'''


'''
References

numpy generate random integer in range
https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.randint.html

scipy.optimize minimize
https://scipy-lectures.org/advanced/mathematical_optimization/

scipy.optimize minimize what method
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

unit testing
https://en.wikipedia.org/wiki/Unit_testing

'''
      
