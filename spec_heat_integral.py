# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 18:30:23 2017

@author: beande
"""

'''
Specific heat integration of the dilution species CO2 and N2 to find the ratio
Y_{N2} / Y_{CO2} for the dilution combustion institute paper
'''
import cantera as ct
import numpy as np

def spec_heat_integral(species, T_f, T_o=298, P=101325, n_points=10000):
    gas = ct.Solution('gri30.cti')
    temps = np.linspace(T_o, T_f, n_points)
    specheats = []
    for i in temps:
        gas.TPX = i, P, '{0}:1'.format(species)
        specheats.append(gas.cp_mass)
    return np.trapz(specheats, temps)

if __name__ == '__main__':
    species = ['N2', 'CO2']
    T_o = 298
    # The ratio of T_CJ/T_o = 12 originates from the ZND model of detonations
    # for more deatil see Lee pg. 78 or Fickett pg. 47
    T_CJ = 12*T_o
    print(spec_heat_integral(species[0], T_CJ, T_o) /
          spec_heat_integral(species[1], T_CJ, T_o))
