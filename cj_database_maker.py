# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 08:42:21 2018

@author: derek

code for making json files that contain cj speeds for specified mixtures
"""

import sd2
import cantera as ct
import numpy as np

def parse_mole_fractions(gas_object):
    mole_fraction_dict = gas.mole_fraction_dict()
    mole_fraction_string_list = ['%s:%s'%(key, mole_fraction_dict[key])
                                                 for key in mole_fraction_dict]
    return (',').join(mole_fraction_string_list)

def create_gas_object(mechanism, temperature, pressure, equivalence_ratio,
                      fuel='CH4', oxidizer='O2'):
    gas = ct.Solution(mechanism)
    gas.TP = temperature, pressure
    gas.set_equivalence_ratio(equivalence_ratio, fuel, oxidizer)
    return gas
    


if __name__ == '__main__':
    
    initial_pressure = ct.one_atm
    
    initial_temperature = 298
    
    mechanism = 'gri30.cti'
    
    equivalence_ratio = np.linspace(0.1, 2.0, 10)
    
    cj_speeds = dict()
    gas_states = dict()
    
    for phi in equivalence_ratio:
        gas = create_gas_object(mechanism, initial_temperature, initial_pressure,
                                phi)
    
        species_mole_fractions = parse_mole_fractions(gas)

    
        cj_speeds[phi], gas_states[phi] = sd2.detonations.calculate_cj_speed(initial_pressure,
                                                           initial_temperature,
                                                           species_mole_fractions,
                                                           mechanism, return_state=True)