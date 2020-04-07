# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 15:50:47 2018

@author: derek
"""

import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import itertools
import sd2
import cantera as ct
import scipy
from pathlib import Path, PurePath


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
    # directory = ['C:\\Users\\derek\\Desktop\\Oxygen_05_29_2019',
    #              'C:\\Users\\derek\\Desktop\\OxyPDE06272019']
    directory = ['C:\\Users\\derek\\Desktop\\q3_data']
    files = []
    
    for folder in directory:
        files.append(Path(folder).glob('**/*.json'))
        
    files = [str(x) for f in files for x in f]
    
    data = list(map(loadPkl, files))
    phis = []
    vels = []
    for group in data:
        for key, item in group.items():
            try:
                vel = np.mean(item['velocity'][0])
            except KeyError:
                vel = 0
            if vel > 100 and  vel < 3000:
                phis.append(item['equivalence_ratio'])
                vels.append(vel)
            else: 
                pass
    
	
    initial_pressure = ct.one_atm
    
    initial_temperature = 298
    
    mechanism = 'gri30_highT.cti'
    
    equivalence_ratio = np.linspace(min(phis), max(phis), 20)
    
    cj_speeds = dict()
    
    for phi in equivalence_ratio:
        try:
            gas = create_gas_object(mechanism, initial_temperature, initial_pressure,
                                    phi)

            species_mole_fractions = parse_mole_fractions(gas)

            cj_speeds_temp = sd2.detonations.calculate_cj_speed(initial_pressure,
                                                               initial_temperature,
                                                               species_mole_fractions,
                                                               mechanism)
            cj_speeds[phi] = cj_speeds_temp
        except ct.CanteraError:
            print('There was a CanteraError')
        except ZeroDivisionError:
            print('There was a zero division error')

#    fit_func = lambda x, a, b, c: a * x ** b + c
#    
#    fit_vals, fit_stats = scipy.optimize.curve_fit(fit_func, phis, vels)

    font_size = 20
    marker_size = font_size-2
    fig, ax = plt.subplots()
    # ax.fill_between(sorted([x[0] for x in cj_speeds.items()]),
    #         sorted([x[1]['cj speed']for x in cj_speeds.items()]),
    #                sorted([0.9*x[1]['cj speed']for x in cj_speeds.items()]), label='90 % Caclulated CJ', alpha=0.25)
    ax.plot(phis, vels, '.', label='Experimental', markersize=marker_size,
            markerfacecolor='none')
    ax.plot(sorted([x[0] for x in cj_speeds.items()]),
            sorted([x[1]['cj speed']for x in cj_speeds.items()]), '--', label='Caclulated CJ',
            markersize=marker_size)
    # ax.plot(phis, [fit_func(*([x] + fit_vals.tolist())) for x in phis], 'o')
    ax.set_ylabel('Detonation Velocity (m/s)', fontsize=font_size)
    ax.set_xlabel('Equivalence Ratio ($\Phi$)', fontsize=font_size)
    ax.tick_params(labelsize=font_size-3)
    ax.legend(fontsize=font_size-6)
    fig.tight_layout()
    fig.show()
            