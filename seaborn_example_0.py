#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 22:23:18 2020

@author: alejandro
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress

# import our own files and reload
import stream_functions
importlib.reload(stream_functions)
import stream_classes
importlib.reload(stream_classes)


# Cargamos nuestra librería estrella

import seaborn as sns


# Generamos algunos números aleatorios con la
# librería numpy

size = 10**3

sim_normal = np.random.normal(loc = 0, scale = 1/2, size = size)
sim_exp = np.random.exponential(scale = 10, size = size)
sim_chi = np.random.chisquare(df = 2, size = size)
sim_t = np.random.standard_t(df = 300, size = size)
sim_gamma = np.random.gamma(shape = 5.5, scale = 10, size = size)


# Para trabajar con seaborn necesitamos tener nuestro datos acomodados en un data frame
df = {'Values' : np.concatenate((sim_normal, sim_exp, sim_chi, sim_t, sim_gamma)), 'Type' : np.repeat(['Normal', 'Exp', 'Chi', 't', 'Gamma'], [size, size, size, size, size])}
df = pd.DataFrame(df)


# Podemos seleccionar el estilo de nuestros futuros plots
sns.set_style("whitegrid")
sns.set_style("darkgrid")
sns.set_style("white")
sns.set_style("dark")
sns.set_style("ticks")

# El de default es
sns.set_theme()


# Tenemos herramientas muy básicas y útiles como un boxplot
sns.boxplot(x = 'Type', y = 'Values', data=df)

# Podemos hacer query's cone l método query de los data frames
df_1 = df.query('Type == "Normal" or Type == "t" or Type == "Chi"')
sns.boxplot(x = 'Type', y = 'Values', data=df_1)
sns.boxplot(x = 'Type', y = 'Values', data=df_1, palette="Paired")
sns.boxplot(x = 'Type', y = 'Values', data=df_1, palette="Set2")
sns.boxplot(x = 'Type', y = 'Values', data=df_1, palette="Set3")
sns.boxplot(y = 'Type', x = 'Values', data=df_1, orient = "h", palette="dark")
sns.boxplot(y = 'Type', x = 'Values', data=df_1, orient = "h", palette="colorblind")
sns.violinplot(y = 'Type', x = 'Values', data=df, orient = "h", palette="dark")
sns.violinplot(y = 'Type', x = 'Values', data=df_1, orient = "h", palette="colorblind")



# Podemos generar también distintos tipos de histogramas

# Create an array with the colors you want to use
colors = ['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0']
# Set your custom color palette
customPalette = sns.set_palette(sns.color_palette(colors))

sns.displot(df, x="Values")
sns.displot(df, x="Values", binwidth=3)
sns.displot(df, x="Values", bins=30)
sns.displot(df, x="Values", hue = 'Type', palette = 'bright')
sns.displot(df, x="Values", hue = 'Type', element = 'step', palette = 'bright')
sns.displot(df, x="Values", hue = 'Type', multiple = 'stack', palette = 'bright')
sns.displot(df_1, x="Values", hue = 'Type', multiple = 'dodge', palette = customPalette)
sns.displot(df_1, x="Values", col = 'Type')

sns.displot(df_1, x="Values", hue = 'Type', multiple = 'stack', palette = 'bright', kind = "kde")
sns.displot(df_1, x="Values", col = 'Type', kind = "kde", fill = True)


# seaborn (junto con matplotlib) nos da la posibilidad de desplegar varios plots en una "matriz" de plots

f = plt.figure(figsize=(10, 8))
gs = f.add_gridspec(1, 2)

with sns.axes_style("darkgrid"):
    ax = f.add_subplot(gs[0, 0])
    sns.boxplot(x = 'Type', y = 'Values', data=df_1, palette="Set3")

with sns.axes_style("whitegrid"):
    ax = f.add_subplot(gs[0, 1])
    sns.violinplot(y = 'Type', x = 'Values', data=df_1, orient = "h", palette=customPalette)

f.tight_layout()




# Cargamos datos reales sincronizandolos
########################################
# input parameters
ric_1 = '^VIX' # MT.AS SAN.MC BBVA.MC REP.MC VWS.CO EQNR.OL MXNUSD=X ^VIX
ric_2 = 'SAN.MC' # MT.AS SAN.MC BBVA.MC REP.MC VWS.CO EQNR.OL MXNUSD=X ^VIX
ric_3 = 'MXNUSD=X' # MT.AS SAN.MC BBVA.MC REP.MC VWS.CO EQNR.OL MXNUSD=X ^VIX
benchmark = '^STOXX' # ^STOXX50E ^STOXX ^S&P500 ^NASDAQ ^FCHI ^GDAXI
file_extension = 'csv'
nb_decimals = 4

# loading data from csv or Excel file
x1, str1, t1 = stream_functions.load_timeseries(ric_1)
x3, str3, t3 = stream_functions.load_timeseries(ric_2)
x4, str4, t4 = stream_functions.load_timeseries(ric_3)
x2, str2, t2 = stream_functions.load_timeseries(benchmark)

# synchronize timestamps
timestamp1 = list(t1['date'].values)
timestamp3 = list(t3['date'].values)
timestamp4 = list(t4['date'].values)
timestamp2 = list(t2['date'].values)
timestamps = list(set(timestamp1) & set(timestamp2) & set(timestamp3) & set(timestamp4))

# synchronised time series for x1 or ric_1
t1_sync = t1[t1['date'].isin(timestamps)]
t1_sync.sort_values(by='date', ascending=True)
t1_sync = t1_sync.reset_index(drop=True)

# synchronised time series for x1 or ric_1
t3_sync = t3[t3['date'].isin(timestamps)]
t3_sync.sort_values(by='date', ascending=True)
t3_sync = t3_sync.reset_index(drop=True)

# synchronised time series for x1 or ric_1
t4_sync = t4[t4['date'].isin(timestamps)]
t4_sync.sort_values(by='date', ascending=True)
t4_sync = t4_sync.reset_index(drop=True)


# synchronised time series for x2 or benchmark
t2_sync = t2[t2['date'].isin(timestamps)]
t2_sync.sort_values(by='date', ascending=True)
t2_sync = t2_sync.reset_index(drop=True)

# table of returns for ric_1 and benchmark
t = pd.DataFrame()
t['date'] = t1_sync['date']
t['price_1'] = t1_sync['close']
t['price_3'] = t3_sync['close']
t['price_4'] = t4_sync['close']
t['price_2'] = t2_sync['close']
t['return_1'] = t1_sync['return_close']
t['return_3'] = t3_sync['return_close']
t['return_4'] = t4_sync['return_close']
t['return_2'] = t2_sync['return_close']

# compute vectors of returns
y = t['return_1'].values
z = t['return_3'].values
w = t['return_4'].values
x = t['return_2'].values


# Formamos un primer Data Frame enfocado en la relación dos a dos de los activos

returns = {'VIX' : y, 'SAN'  : z, 'MXNUSD' : w, 'STOXX' : x}
returns = pd.DataFrame(returns)

sns.jointplot(data = returns, x="STOXX", y="VIX")
sns.jointplot(data = returns, x="STOXX", y="VIX", kind = "reg")

sns.relplot(data=returns, x="STOXX", y="VIX")
sns.rugplot(data=returns, x="STOXX", y="VIX")

sns.regplot(data=returns, x="STOXX", y="VIX")
sns.lmplot(data=returns, x="STOXX", y="VIX")

