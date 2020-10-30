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
colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
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
ric_4 = 'VWS.CO' # MT.AS SAN.MC BBVA.MC REP.MC VWS.CO EQNR.OL MXNUSD=X ^VIX
benchmark = '^STOXX' # ^STOXX50E ^STOXX ^S&P500 ^NASDAQ ^FCHI ^GDAXI
file_extension = 'csv'
nb_decimals = 4

# loading data from csv or Excel file
x1, str1, t1 = stream_functions.load_timeseries(ric_1)
x3, str3, t3 = stream_functions.load_timeseries(ric_2)
x4, str4, t4 = stream_functions.load_timeseries(ric_3)
x5, str5, t5 = stream_functions.load_timeseries(ric_4)
x2, str2, t2 = stream_functions.load_timeseries(benchmark)

# synchronize timestamps
timestamp1 = list(t1['date'].values)
timestamp3 = list(t3['date'].values)
timestamp4 = list(t4['date'].values)
timestamp5 = list(t5['date'].values)
timestamp2 = list(t2['date'].values)
timestamps = list(set(timestamp1) & set(timestamp2) & set(timestamp3) & set(timestamp4) & set(timestamp5))

# synchronised time series for x1 or ric_1
t1_sync = t1[t1['date'].isin(timestamps)]
t1_sync.sort_values(by='date', ascending=True)
t1_sync = t1_sync.reset_index(drop=True)

# synchronised time series for x2 or ric_2
t3_sync = t3[t3['date'].isin(timestamps)]
t3_sync.sort_values(by='date', ascending=True)
t3_sync = t3_sync.reset_index(drop=True)

# synchronised time series for x3 or ric_3
t4_sync = t4[t4['date'].isin(timestamps)]
t4_sync.sort_values(by='date', ascending=True)
t4_sync = t4_sync.reset_index(drop=True)

t5_sync = t5[t5['date'].isin(timestamps)]
t5_sync.sort_values(by='date', ascending=True)
t5_sync = t5_sync.reset_index(drop=True)


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
t['price_5'] = t5_sync['close']
t['price_2'] = t2_sync['close']
t['return_1'] = t1_sync['return_close']
t['return_3'] = t3_sync['return_close']
t['return_4'] = t4_sync['return_close']
t['return_5'] = t5_sync['return_close']
t['return_2'] = t2_sync['return_close']

# compute vectors of returns
y = t['return_1'].values
z = t['return_3'].values
w = t['return_4'].values
v = t['return_5'].values
x = t['return_2'].values


time_series = t.filter(items=['date', 'price_1', 'price_2', 'price_3',
                              'price_4', 'price_5'])
sns.lineplot(data=time_series, palette="tab10", linewidth=1)

# Formamos un primer Data Frame enfocado en la relación dos a dos de los activos

returns = {'VIX' : y, 'SAN'  : z, 'MXNUSD' : w, 'VWS' : v, 'STOXX' : x}
returns = pd.DataFrame(returns)


# Reiniciamos los parámetros de colores 
sns.set(color_codes=False)
##
sns.set()
##
sns.reset_orig()
##
sns.color_palette(palette=None)
sns.set(color_codes=True)

sns.jointplot(data = returns, x="STOXX", y="VIX")
sns.jointplot(data = returns, x="STOXX", y="VIX", kind = "reg", color="b")

sns.relplot(data=returns, x="STOXX", y="VIX")
sns.rugplot(data=returns, x="STOXX", y="VIX")

sns.regplot(data=returns, x="STOXX", y="VIX")
sns.lmplot(data=returns, x="STOXX", y="VIX")

g = sns.JointGrid(data=returns, x='STOXX', y="MXNUSD")
g.plot_joint(sns.histplot)
g.plot_marginals(sns.boxplot)

sns.pairplot(data = returns)

g = sns.PairGrid(returns)
g.map_upper(sns.rugplot)
g.map_upper(sns.histplot)
g.map_lower(sns.kdeplot, fill=True)
g.map_diag(sns.histplot, kde=True)


colors = ['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494']
customPalette = sns.color_palette(colors, as_cmap=True)
corrMatrix = returns.corr()

sns.heatmap(corrMatrix, vmax=.3, square=True)

mask = np.zeros_like(corrMatrix)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.heatmap(corrMatrix, mask=mask, vmax=.3, square=True, 
                     cmap = customPalette)

# Formamos un nuevo Data Frame para emplorar relaciones en conjunto
rics = {'Benchmark' : np.concatenate((x, x, x, x)), 'rics returns' : np.concatenate((y, z, w, v)),
        'rics names' : np.repeat(['VIX', 'SAN', 'MXNUSD', 'VWS'], [len(x), len(x), len(x), len(x)]), 
        'market'  : np.repeat(['American', 'European', 'American', 'European'], [len(x), len(x), len(x), len(x)])}
rics = pd.DataFrame(rics)


g = sns.PairGrid(rics, hue="rics names", corner=True)
g.map_lower(sns.kdeplot, hue=None, levels=5, color=".2")
g.map_lower(sns.scatterplot, marker="+")
g.map_diag(sns.histplot, element="step", linewidth=0, kde=True)
g.add_legend(frameon=True)
g.legend.set_bbox_to_anchor((.7, .7))

sns.lmplot(x="Benchmark", y="rics returns", hue="rics names", data=rics)
sns.pairplot(data=rics, hue="rics names")

sns.lmplot(x="Benchmark", y="rics returns", hue="market", data=rics)
sns.pairplot(data=rics, hue="market")


sns.lmplot(x="Benchmark", y="rics returns", hue="rics names", col="market", data=rics)
sns.lmplot(x="Benchmark", y="rics returns", col="rics names", data=rics,
           col_wrap=2, height=3)
# Create an array with the colors you want to use
colors = ['#1b9e77','#d95f02','#7570b3','#e7298a']
# Set your custom color palette
customPalette = sns.set_palette(sns.color_palette(colors))

sns.lmplot(x="Benchmark", y="rics returns", hue="rics names", data=rics, palette=customPalette)
plt.title("A Title") # Add plot title
plt.ylabel("") # Adjust the label of the y-axis
plt.xlabel(benchmark)  # Adjust the label of the x-axis
plt.ylim(-0.4, 0.6) # Adjust the limits of the y-axis
# plt.xlim(0,10) # Adjust the limits of the x-axis
# plt.legend(loc=3)
# plt.legend([],[], frameon=False)
plt.tight_layout() 
# plt.show() # Show the plot
#plt.savefig("/home/alejandro/mi_plot.jpeg") # Save the plot as a figure
plt.savefig("/home/alejandro/mi_plot_1.png", transparent=False)  # Save transparent figure
 