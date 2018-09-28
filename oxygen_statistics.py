# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 15:50:47 2018

@author: derek
"""

import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import sd2
import cantera as ct


def loadPkl(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)
		
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
    
    


if __name__ == "__main__":
    directory = ['C:\\Users\\derek\\Desktop\\8_21_2018\\',
                 'C:\\Users\\derek\\Desktop\\8_28_2018\\',
                 'C:\\Users\\derek\\Desktop\\8_28_2018\\']
    files = itertools.chain(*[glob.glob(x + '*.json') for x in directory])
    
    
    data = list(map(loadPkl, files))
    phis = []
    vels = []
    for group in data:
        for key, item in group.items():
            vel = np.mean(item['velocity'][0])
            if vel > 100 and  vel < 3000:
                phis.append(item['equivalence_ratio'])
                vels.append(vel)
            else: 
                pass
    
	
    initial_pressure = ct.one_atm
    
    initial_temperature = 298
    
    mechanism = 'gri30.cti'
    
    equivalence_ratio = np.linspace(min(phis), max(phis), 15)
    
    cj_speeds = dict()
    
    for phi in equivalence_ratio:
        gas = create_gas_object(mechanism, initial_temperature, initial_pressure,
                                phi)
    
        species_mole_fractions = parse_mole_fractions(gas)
    
        cj_speeds[phi] = sd2.detonations.calculate_cj_speed(initial_pressure,
                                                           initial_temperature,
                                                           species_mole_fractions,
                                                           mechanism)
	
    font_size = 26
    fig, ax = plt.subplots()
    ax.plot(phis, vels, 'xk')
    ax.plot(*zip(*sorted(cj_speeds.items())), 'ob')
    ax.plot()
    ax.set
    ax.set_ylabel('Velocity (m/s)', fontsize=font_size)
    ax.set_xlabel('$\Phi$', fontsize=font_size)
    ax.tick_params(labelsize=font_size-3)
    fig.show()
            