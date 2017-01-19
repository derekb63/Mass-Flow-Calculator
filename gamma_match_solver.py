# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 16:10:15 2017

@author: beande
"""

from sympy.solvers import solve
from sympy import Symbol
import cantera as ct


def gamma_finder(comp):
    T = 298
    P = 101325
    gas = ct.Solution('gri30.cti')
    gas.TPX = T, P, comp
    k = gas.cp_mass/gas.cv_mass
    return k

phi = 1
elements = ['C3H8', 'N2O', 'N2', 'CO2']
coeffs = [1, 10*phi, 1, None]

ct_comp = [x + ':1' for x in elements]
k = [gamma_finder(x) for x in ct_comp]
props = dict(zip(elements, zip(k, coeffs)))

N_CO2 = Symbol('N_CO2')

solve(1/(sum(coeffs)))*(N_C3H8*gamma['C3H8']+N_N2O*gamma[']), N_CO2)
