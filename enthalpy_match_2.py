#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:06:52 2017

@author: aero-10
"""

'''
Code to match the enthalpy between two diluents in a presribed fuel oxidizer
mixture
'''

import numpy as np
import cantera as ct



if __name__ == '__main__':
    Fuel = 'C3H8'
    Oxidixer = 'N2O'
    Diluent_1 = 'N2'
    Diluent_2 = 'CO2'