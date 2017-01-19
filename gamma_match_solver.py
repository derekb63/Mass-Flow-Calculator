# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 16:10:15 2017

@author: beande
"""

from sympy.solvers import solve
from sympy import Symbol
import cantera as ct
import numpy as np
import sys


def gamma_finder(comp):
    T = 298
    P = 101325
    gas = ct.Solution('gri30.cti')
    gas.TPX = T, P, comp
    k = gas.cp_mass/gas.cv_mass
    return k

phi = 1
N2_dilution = 1
fuel = 'C3H8'
oxidizer = 'N2O'
known_diluent = 'N2'
unknown_diluent = 'CO2'
elements = [fuel, oxidizer, known_diluent, unknown_diluent]
if len(set(elements)) != len(elements):
    sys.exit('There are duplicates in the elments list.' +
             'This is not allowed' +
             'Please eliminate the duplicate and try again')
coeffs = [1, 10*phi, N2_dilution, None]

ct_comp = [x + ':1' for x in elements]
k = [gamma_finder(x) for x in ct_comp]
props = dict(zip(elements, zip(k, coeffs)))

X = Symbol(elements[-1])

a = solve(1/(sum(coeffs[0:-1])) *
          sum([np.prod(props[x]) for x in elements[0:-1]]) -
             1/(sum(coeffs[0:2]) + X) *
               (sum([np.prod(props[x]) for x in elements[0:2]]) +
                X*props[elements[-1]][0]), X)
a = a[0]
print(a)
