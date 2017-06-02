#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 2017

@author: beande

Uses python 2.7
"""

import cantera as ct
from SDToolbox import CJspeed, PostShock_eq
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from multiprocessing import Process, Queue
import time
import re


"""
Chemical equilibrium calculations to find the exhaust product concentrations
of diluted detonations
"""


def equilibrium(comp, P=101325, T=298, mech='gri30.cti', q = Queue()):
    [speed, R] = CJspeed(P, T, comp, mech, 0)
    gas = PostShock_eq(speed, P, T, comp, mech)
#    print tabulate([['For gas composition'],
#                    ['{0}'.format(comp)],
#                    [None],
#                    ['Speed', round(speed, 0), 'm/s'],
#                    ['Temperature', round(gas.T, 0), 'K'],
#                    ['Pressure Ratio', round(gas.P/ct.one_atm, 0)],
#                    [None],
#                    ['End Table'],
#                    [None]],
#                   headers=['Variable', 'Value', 'Units'])
    result = {'Temp': gas.T, 'Press_Ratio': gas.P/ct.one_atm,
              'Comp': comp, 'Speed': speed}
    q.put(result)
    return gas


if __name__ == '__main__':
    mech = ['gri30_highT.cti']
#    mechanism = ['gri30_highT.cti', 'gri30.cti', 'CSMmech7_2.cti']
    P = 101325
    T = 298
    Propane = 1
    Nitrous = np.linspace(1, 13, num=30)
    comp = 'C3H8:1 N2O:10'
    diluent = np.linspace(0, 10, num=20)
#    for mech in mechanism:
    q = Queue()
    results = []
    processes = []
    for i in Nitrous:
        comp = 'C3H8:{0} N2O:{1} CO2:{2}'.format(Propane, round(i, 2), 0)
        p = Process(target=equilibrium, args=(comp, P, T, mech, q))
        processes.append(p)
        p.start()
    [results.append(q.get()) for i in processes]
    [i.join() for i in processes]
    oxidizer = []
    vel = []
    for i in results:
        oxidizer.append(float(re.findall(r'\d*\.\d+|d\+', i['Comp'])[0]))
        vel.append(i['Speed'])
    plt.plot(oxidizer, vel, 'x', label=mech)
    plt.xlabel(r'$N_{2}O $ moles, $C_{3}H_{8}:1$')
    plt.ylabel(r'Velocity (m/s)')
    plt.legend()
