# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 12:36:23 2017

@author: beande
"""

# Enthalpy matching function

import cantera as ct
import matplotlib.pyplot as plt

T = 298
P = 101325
gas = ct.Solution('gri30.cti')
comp = 'C3H8:1, N2O:10, N2:2'
gas.TPX = T, P, comp
k_N2 = gas.cp_mass/gas.cv_mass

k_CO2 = 0.0
k_m = []
i = 0
i_m = []
gas2 = ct.Solution('gri30.cti')
while k_CO2 < 0.999*k_N2 or k_CO2 > 1.0001*k_N2:
    comp2 = 'C3H8:1, N2O:10, CO2:{0}'.format(i)
    gas2.TPX = T, P, comp2
    k_CO2 = gas2.cp_mass/gas2.cv_mass
    k_m.append(k_CO2)
    i_m.append(i)
    if k_CO2 < k_N2:
        i += 0.001
    else:
        i -= 0.001
    if i < 0:
        break
print(k_CO2)
plt.plot(i_m, k_m)
