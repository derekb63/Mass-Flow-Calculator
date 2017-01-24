#! usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 19:32:20 2016

@author: beande
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cantera as ct
from nptdms import TdmsFile
from tkinter import *
from tkinter.filedialog import askopenfilename
import time
from functions import reformat, find_M_dot, velocity_calc, FindFile


def mass_flow_calc(diluent, Tname=None, Pname=None, PDname=None):
    start = time.time()

    # Constants

    # diluent = 'CO2'
    fuel = 'C3H8'
    oxidizer = 'N2O'
    Gases = [oxidizer, fuel, diluent]  # Species of the gas used ct form
    R = ct.gas_constant / 1000       # Gas constant (kPa m^3/kmol-K)
    P_d = 101325                     # Downstream pressure in kPa

    """ If you don't want to choose the files """
    # Tname  = 'January20\TC.tdms'
    # Pname  = 'January20\PT.tdms'
    # PDname = 'January20\PD.tdms'

    # Pressure transducer calibrations
    cals = [[31230, -125.56],
            [31184, -125.54],
            [15671, -62.619],
            [15671, -62.619],
            [15671, -62.616],
            [15671, -62.616],
            [15667, -62.535],
            [15642, -62.406]]
    # The number of transducers and thermocouples read in the tdms file
    # numPT = 8
    # numTC = 3
    numPT = 8
    numTC = 3
    gas = ct.Solution('gri30.xml')


    # Import Temperature File
#    try:
#        Tname
#    except NameError:
#        Tname = FindFile('Temperature Open')                  
#    # Import Pressure File
#    try:
#        Pname
#    except NameError:
#        Pname = FindFile('Pressure Open')
#    # Import Photodiode File
#    try:
#        PDname
#    except NameError:
#        PDname = FindFile('Photodiode Open')
        
    if Tname == None:
        Tname = FindFile('Temperature Open')
    if Pname == None:
        Pname = FindFile('Pressure Open')
    if PDname == None:
        PDname = FindFile('Photodiode Open')


    ################################################

    Pressfile = TdmsFile(Pname)
    Pressdata = Pressfile.as_dataframe(time_index=True, absolute_time=False)
    Pressdata = reformat(Pressdata)

    Tempfile = TdmsFile(Tname)
    Tempdata = Tempfile.as_dataframe(time_index=True, absolute_time=False)
    Tempdata = reformat(Tempdata)
    numTests = len(Tempdata)
    ##############################################################

    ##############################################################
    # Example to run to find m_dot
    '''
    TC=1
    D_orifice=0.047 ##Diameter of the orifice in INCHES
    Gas='C3H8'
    test=1
    ducer=6
    m_dot=find_M_dot(Tempdata,Pressdata,test, ducer, TC, D_orifice, cals, Gas)
    print(type(m_dot))
    print(Gas,test)
    print(m_dot)
    print()

    '''

    # Initialize M_dot
    M_dot = pd.DataFrame(index=range(0, numTests), columns=Gases)

    for Gas in Gases:

        if (Gas == 'Propane') or (Gas == 'C3H8'):
            ducer = 6
            TC = 1
            D_orifice = 0.047  # Diameter of the orifice in INCHES
        elif (Gas == 'NitrousOxide') or (Gas == 'N2O'):
            ducer = 5  # On PDE ducer = 5 changed to 6 for testing
            D_orifice = 0.142  # Diameter of the orifice in INCHES
            TC = 2
        elif (Gas == 'Nitrogen') or (Gas == 'N2'):
            ducer = 7
            D_orifice = 0.063  # Diameter of the orifice in INCHES
            TC = 3
        elif (Gas == 'CO2') or (Gas == 'CarbonDioxide'):
            ducer = 7
            D_orifice = 0.063  # Diameter of the orifice in INCHES
            TC = 3
        else:
            print('Gas Not Recognized')

        for test in range(len(Pressdata)):

            m_dot = find_M_dot(Tempdata, Pressdata, test, ducer, TC,
                               D_orifice, cals, Gas)

            M_dot[Gas][test] = m_dot

    # Equivelance Ratio
    phi = 10*np.divide(M_dot[fuel], M_dot[oxidizer]).rename('Phi')

    print()
    # Mass Dilution Ratio
    dilution = np.divide(M_dot[diluent],
                         M_dot[fuel] + M_dot[oxidizer] +
                         M_dot[diluent]).rename('Diluent'+' ('+diluent+')')

    # Place the equivalence ratio and dilution data into a pandas.DataFrame
    # so that it can be easily managed/saved later

    Data = pd.concat([phi, dilution], axis=1)

    del Pressdata, Tempdata, Pressfile, Tempfile, M_dot, m_dot, dilution, phi

    # Put the velocity and error data into the Data pandas.DataFrame
    # The necessary data is now all in the same place

    Data = pd.concat([Data, velocity_calc(PDname, method='max')], axis=1)

    Data.index.name = 'Test Number'

    # Write the data file to the same location as the photodiode data was
    # sourced by creating a new file or appending to the file

    Data.to_csv('/'.join(PDname.split('/')[:-1]) + '/testdata.csv', mode='a')

    print(Data)

    end = time.time()
    print('Run Time:', end-start, 'seconds')

    return Data

if __name__ == '__main__':
    Data = mass_flow_calc(diluent = 'N2')
