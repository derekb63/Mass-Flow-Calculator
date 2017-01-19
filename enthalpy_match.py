# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 12:36:23 2017

@author: beande
"""

# Enthalpy matching function

import cantera as ct
from SDToolbox import CJspeed
from scipy.optimize import fsolve

import time


def CJ_func(comp, A):
    [Vel, _] = CJspeed(101325, 298, comp, 'gri30.cti', 0)
    print(Vel, comp)
    A.append(Vel)
    return A, Vel

start = time.time()
T = 298
P = 101325
gas = ct.Solution('gri30.cti')
j = 1
comp = 'C3H8:1, N2O:10, N2:{0}'.format(j)
gas.TPX = T, P, comp
k_N2 = gas.cp_mass/gas.cv_mass

k_CO2 = 0.0
k_m = []
i = 0
i_m = []
gas2 = ct.Solution('gri30.cti')

gas_check = ct.Solution('gri30.cti')
gas_check.TPX = T, P, 'CO2:1'
if k_N2 >  gas_check.cp_mass/gas_check.cv_mass:
    print('This will not work. The target gamma is too high')
else:
    while k_CO2 < 0.9999*k_N2 or k_CO2 > 1.0001*k_N2:
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
            i = 0
            break
print('k_N2 :', k_N2)
print('k_CO2:', k_CO2)
print('Error:', ((k_N2-k_CO2)/k_N2)*100)

A = []
B = []
comp1 = 'C3H8:1 N2O:10 N2:{0}'.format(j)
comp2 = 'C3H8:1 N2O:10 CO2:{0}'.format(1/i)
CJ_func(comp1, A)
CJ_func(comp2, B)
end = time.time()
print('Time :', end-start)
print('CO2  :', round(i, 6))
print('N2   :', round(j, 6))
