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


"""
Chemical equilibrium calculations to find the exhaust product concentrations
of diluted detonations
"""



def equilibrium(comp, P=101325, T=298, mech='gri30.cti'):
    [speed, R] = CJspeed(P, T, comp, mech, 0)
    gas = PostShock_eq(speed, P, T, comp, mech)
    print 'For gas composition {0} the following values were caclulated'\
          .format(comp)
    print ''
    print tabulate([['Speed', round(speed, 0), 'm/s'],
                    ['Temperature', round(gas.T, 0), 'K'],
                    ['Pressure Ratio', round(gas.P/ct.one_atm, 0)],
                    [None],
                    ['End Table']],
                   headers=['Variable', 'Value', 'Units'])
#    data = open('outputdata', 'w')
#    data.write(gas())
#    data.close(0)
    return gas


if __name__ == '__main__':
    mech = 'gri30_highT.cti'
    P = 101325
    T = 298
    comp = 'C3H8:1 N2O:10'
    one = Process(target=equilibrium, args=(comp, P, T, mech))
    comp2 = 'C3H8:2 N2O:10 CO2:3'
    two = Process(target=equilibrium, args=(comp2, P, T, mech))
    one.start()
    two.start()
    one.join()
    two.join()    
