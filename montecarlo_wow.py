# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:45:56 2021

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

# inputs
str_name = 'WoW-Frostbolt' 
spell = 1085
cast_time = 1.7 # seconds
total_time = 300 # seconds
critical = 0.21 # pct chance
nb_sims = 3*10**2

# Theoretical values for a Bernoulli distribution
expected_spell_damage = critical*(2*spell) + (1-critical)*spell
expected_spell_dps = expected_spell_damage / cast_time

# Montecarlo simulation: generate random variables Bernoulli
nb_spells = int(total_time / cast_time)
mtx_unif = np.random.uniform(low=0.0, high=1.0, size=[nb_spells,nb_sims])
mtx_montecarlo = spell*(mtx_unif > critical) + 2*spell*(mtx_unif <= critical)
vec_dps = sum(mtx_montecarlo)/total_time

# analyse distribution
jb = stream_classes.jarque_bera_test(str_name)
jb.size = len(vec_dps)
jb.returns = vec_dps
jb.str_name = str_name
jb.compute()
jb.plot_histogram()
print(jb)
print('-----')

# adjust theoretical Bernoulli
expected_spell_dps = expected_spell_damage * nb_spells / total_time

# print
conf_radius = 1.96 * jb.std / np.sqrt(jb.size)
q_left = np.percentile(vec_dps,2.5)
q_right = np.percentile(vec_dps,97.5)
print('Distribution min/max dps is ' + str(np.min(vec_dps)) + ' / ' + str(np.max(vec_dps)))
print('Distribution confidence interval 95% for dps is (' \
      + str(q_left) + ', ' + str(q_right) + ')')
print('Assuming normality, the confidence interval 95% for dps is (' \
      + str(jb.mean-1.96*jb.std) + ', ' + str(jb.mean+1.96*jb.std) + ')')
print('-----')
print('Mean confidence interval 95% for dps is (' \
      + str(jb.mean-conf_radius) + ', ' + str(jb.mean+conf_radius) + ')')
print('Expected spell damage (dps) is ' + str(expected_spell_damage)\
      + ' (' + str(expected_spell_dps) + ')')
print('-----')






# expected_spell_dps= expected_spell_damage * nb_spells / total_time