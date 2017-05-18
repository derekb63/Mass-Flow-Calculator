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
from multiprocessing import Process, Pool, TimeoutError
from threading import Thread


"""
Chemical equilibrium calculations to find the exhaust product concentrations
of diluted detonations
"""


def equilibrium(comp, P=101325, T=298, mech='gri30.cti'):
    [speed, R] = CJspeed(P, T, comp, mech, 0)
    gas = PostShock_eq(speed, P, T, comp, mech)
    print tabulate([['For gas composition'],
                    ['{0}'.format(comp)],
                    [None],
                    ['Speed', round(speed, 0), 'm/s'],
                    ['Temperature', round(gas.T, 0), 'K'],
                    ['Pressure Ratio', round(gas.P/ct.one_atm, 0)],
                    [None],
                    ['End Table'],
                    [None]],
                   headers=['Variable', 'Value', 'Units'])
    return gas


if __name__ == '__main__':
    mech = 'gri30_highT.cti'
    P = 101325
    T = 298
    comp = 'C3H8:1 N2O:10'
    diluent = np.linspace(0, 10, num=20)
    thread_list = []
    manager = Manager()
    return_dict = manager.dict()
    for i in diluent:
        comp = 'C3H8:1 N2O:10 CO2:{0}'.format(round(i, 2))
#        worker = Thread(target=equilibrium, args=(comp, P, T, mech))
#        worker.setDaemon(True)
#        worker.start()
        thread_list.append(Process(target=equilibrium, args=(comp, P, T, mech)))
    [i.start() for i in thread_list]
    [j.join() for j in thread_list]
