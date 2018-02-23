#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:48:47 2017

@author: aero-10
"""

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sci_stats

co2_data = pd.DataFrame.from_csv('CO2_output_raw.csv')
n2_data = pd.DataFrame.from_csv('N2_output_raw.csv')

cj_data = pd.DataFrame.from_csv('data.csv')


co2_exp = [co2_data['Diluent (CO2) mean'], co2_data['V mean']/1920]
n2_exp = [n2_data['Diluent (N2) mean'].values, n2_data['V mean'].values/1920]

exp_fit_co2 = sci_stats.linregress(co2_exp[0], co2_exp[1])
exp_fit_n2 = sci_stats.linregress(n2_exp[0], n2_exp[1])

co2_cj = [cj_data[' Y_co2'].values, cj_data[' CJ_co2'].values/2188.749]
n2_cj = [cj_data[' Y_n2'].values, cj_data[' CJ_n2'].values/2188.749]

cj_fit_co2 = sci_stats.linregress(co2_cj[0], co2_cj[1])
cj_fit_n2 = sci_stats.linregress(co2_cj[0], n2_cj[1])

co2_fit = [-2052.15, 2117.01]
n2_fit = [-188.54, 1895.56]

print(exp_fit_co2)
print()
print(cj_fit_co2)
print()
print(exp_fit_n2)
print()
print(cj_fit_n2)
#plt.plot(co2_exp[0], co2_exp[1], 'x')
#plt.plot(n2_exp[0], n2_exp[1], 'o')
plt.figure()
plt.plot(co2_cj[0], co2_cj[1], '--', label='CJ CO2')
plt.plot(n2_cj[0], n2_cj[1], '--', label='CJ N2')
plt.plot(co2_cj[0], (co2_fit[0]*co2_cj[0]+co2_fit[1])/co2_fit[1], label='Exp CO2' )
plt.plot(co2_cj[0]*1.37, (co2_fit[0]*co2_cj[0]+co2_fit[1])/co2_fit[1], label=r'Exp CO2 h & $\gamma$ match' )
plt.plot(co2_cj[0]*1.05, (co2_fit[0]*co2_cj[0]+co2_fit[1])/co2_fit[1], label='Exp CO2 h match' )
plt.plot(n2_cj[0], (n2_fit[0]*n2_cj[0]+n2_fit[1])/n2_fit[1], label='Exp N2')
plt.xlim([0, 0.40])
plt.ylim([0.6, 1.01])
plt.legend()
plt.xlabel(r'$Y_{diluent}$')
plt.ylabel(r'$\frac{V}{V_{CJ}}$', rotation=0)

