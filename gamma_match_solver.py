#! usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 16:10:15 2017

@author: beande

This python code caclulates the requred concetration of an unknown diluent
based on the number of moles used to dilute the fuel and oxidizer mixture.
Then the CJ detonation velocity is calculated for both mixtures for comparision
using the SDToolbox
"""

from sympy.solvers import solve
from sympy import Symbol
import cantera as ct
from SDToolbox import CJspeed
import numpy as np
import sys


def gamma_finder(comp, T=298, P=101325):
    # Determine the ratio of specific heats for the input composition
    # The input must be in the cantera mole fraction format
    # Example input 'C3H8:1 N2O:10 N2:1 CO2:8'
    gas = ct.Solution('gri30.cti')
    gas.TPX = T, P, comp
    k = gas.cp_mass/gas.cv_mass
    return k


def CJ_func(comp, T=298, P=101325, vel_list=[]):
    # Find the CJ velocity for the input composition
    # The input must be in the cantera mole fraction format
    # Example input 'C3H8:1 N2O:10 N2:1 CO2:8'
    [Vel, _] = CJspeed(P, T, comp, 'gri30.cti', 0)
    print(Vel, comp)
    vel_list.append(Vel)
    return vel_list


def energy_release(comp, T=298, P=101325):
    # Caclulate the heat release of the mixture if allowed to go to
    # equilibrium
    gas = ct.Solution('gri30.cti')
    gas.TPX = T, P, comp
    h_react = gas.enthalpy_mole
    gas.equilibrate('HP')
    h_prod = gas.enthalpy_mole
    return h_react - h_prod


def Ideal_CJ(comp, T=298, P=101325):
    # calculate the ideal CJ velocity for the mixture based on the ideal CJ
    # detonation equation
    q = np.abs(energy_release(comp, T, P))
    print(q)
    gamma = gamma_finder(comp, T, P)
    D_CJ = np.sqrt(2*((gamma**2)-1)*q)
    return D_CJ


phi = 1
known_dilution = 1
fuel = 'C3H8'
oxidizer = 'N2O'
F_O_stoic = 10
known_diluent = 'N2'
unknown_diluent = 'CO2'
elements = [fuel, oxidizer, known_diluent, unknown_diluent]

# If there are multiples of compound in the input the solver will not work
# so the code is terminated if multiples are found

if len(set(elements)) != len(elements):
    sys.exit('There are duplicates in the elments list.' +
             'This is not allowed' +
             'Please eliminate the duplicate and try again')

coeffs = [1, F_O_stoic*phi, known_dilution, None]

# Get the ratio of specific heats for the constituents and place them in a dict
# for refernce later in the code

ct_comp = [x + ':1' for x in elements]
k = [gamma_finder(x) for x in ct_comp]
props = dict(zip(elements, zip(k, coeffs)))

# Assign the sympy solver variable X as the diluent with unknown concentration

X = Symbol(elements[-1])

# Use the sympy solver to find the requred number of moles of the unknown
# diluent. The process to get this equation can be found in the supporting
# LaTeX document

a = solve(1/(sum(coeffs[0:-1])) *
          sum([np.prod(props[x]) for x in elements if x != unknown_diluent]) -
          1/(sum(coeffs[0:2]) + X) *
          (sum([np.prod(props[x]) for x in elements[0:-2]]) +
          X*props[elements[-1]][0]), X)

coeffs[-1] = a[0]

# Reassign the dictionary to include the caclulated number of moles for the
# unknown diluent

props = dict(zip(elements, zip(k, coeffs)))

# Define the inputs for the CJ Detionation calulations for each diluent

CJ_comp1 = ''.join([x + ':' + str(props[x][1]) + ' '
                    for x in elements if x != unknown_diluent])

CJ_comp2 = ''.join([x + ':' + str(props[x][1]) + ' '
                    for x in elements if x != known_diluent])

vel = CJ_func(CJ_comp1)
CJ_func(CJ_comp2)

print(vel)
