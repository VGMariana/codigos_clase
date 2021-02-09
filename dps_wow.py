# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:24:04 2021

@author: Meva
"""

# import libraries
import numpy as np

# simulation 1
mean_1 = 4863.3
std_1 = 202
nb_sim = 26083

# simulation 2
mean_2 = 4847
std_2 = 211.5

# confidence intervals
conf_radius_1 = 1.96*std_1/np.sqrt(nb_sim)
conf_radius_2 = 1.96*std_2/np.sqrt(nb_sim)
conf_int_1 = mean_1 + np.array([-1,1])*conf_radius_1
conf_int_2 = mean_2 + np.array([-1,1])*conf_radius_2

# who wins?
print('-----')
if conf_int_1[1] < conf_int_2[0]:
    print('Simulation 2 wins')
elif conf_int_2[1] < conf_int_1[0]:
    print('Simulation 1 wins')
else:
    print('Cannot determine a winner, the confidence intervals intersect')

# print details on confidence intervals
print('-----')
print('Simulation 1')
print('confidence radius 1 = ' + str(conf_radius_1))
print('confidence interval 1:')
print(conf_int_1)
print('-----')
print('Simulation 2')
print('confidence radius 2 = ' + str(conf_radius_2))
print('confidence interval 2:')
print(conf_int_2)