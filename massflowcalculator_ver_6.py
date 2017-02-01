#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 19:32:20 2016

@author: beande
"""
import numpy as np
import pandas as pd
from nptdms import TdmsFile
import time
from functions import reformat, find_M_dot, velocity_calc, FindFile
from functions import Fuel_Oxidizer_Ratio


def mass_flow_calc(fuel='C3H8', oxidizer='N2O', diluent=None,
                   Tname=None, Pname=None, PDname=None, save=True,
                   method='max'):
    start = time.time()

    Gases = [oxidizer, fuel, diluent]  # Species of the gas used ct form

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

    # Get the filepaths for the termperature, pressure and photodiode data if
    # they are not given in the function call
    if Tname is None:
        Tname = FindFile('Temperature Open')

    if Pname is None:
        Pname = FindFile('Pressure Open')

    if PDname is None:
        PDname = FindFile('Photodiode Open')

    ################################################
    # Import the TDMS files corresponding to the filepaths
    Pressfile = TdmsFile(Pname)
    Pressdata = Pressfile.as_dataframe(time_index=True, absolute_time=False)
    Pressdata = reformat(Pressdata)

    Tempfile = TdmsFile(Tname)
    Tempdata = Tempfile.as_dataframe(time_index=True, absolute_time=False)
    Tempdata = reformat(Tempdata)

    numTests = len(Tempdata)
    ##############################################################

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
    phi = np.divide(np.divide(M_dot[fuel], M_dot[oxidizer]).rename('Phi'),
                    Fuel_Oxidizer_Ratio(fuel, oxidizer))

    # Mass Fraction of Diluent
    dilution = np.divide(M_dot[diluent],
                         M_dot[fuel] + M_dot[oxidizer] +
                         M_dot[diluent]).rename('Diluent'+' ('+diluent+')')

    # Place the equivalence ratio and dilution data into a pandas.DataFrame
    # so that it can be easily managed/saved later

    Data = pd.concat([phi, dilution], axis=1)

    del Pressdata, Tempdata, Pressfile, Tempfile, M_dot, m_dot, dilution, phi

    # Put the velocity and error data into the Data pandas.DataFrame
    # The necessary data is now all in the same place

    Data = pd.concat([Data, velocity_calc(PDname, method)], axis=1)

    Data.index.name = 'Test Number'

    # Write the data file to the same location as the photodiode data was
    # sourced by creating a new file or appending to the file
    if save is True:
        Data.to_csv('/'.join(PDname.split('/')[:-1]) +
                    'testdata.csv', mode='a')

    print(Data)
    Data.plot(x='Phi', y=['V1'], marker='x', linestyle='None',
              ylim=(500, 3500), xlim=[0.95,1.05])

    end = time.time()
    print('Run Time:', end-start, 'seconds')

    return Data

if __name__ == '__main__':

#    Data = mass_flow_calc(diluent='N2',
#                          Tname='D:/PDE Project/Dilution Project/' +
#                          'Dilution Experiment Tests/Phase 1/' +
#                         'January 27/No Dilution/TC.tdms',
#                         Pname='D:/PDE Project/Dilution Project/' +
#                         'Dilution Experiment Tests/Phase 1/' +
#                         'January 27/No Dilution/PT.tdms',
#                         PDname='D:/PDE Project/Dilution Project/' +
#                         'Dilution Experiment Tests/Phase 1/' +
#                         'January 27/No Dilution/PD.tdms', save = False)
    Data = mass_flow_calc(diluent='N2', save=False, method='max')
    Data = mass_flow_calc(diluent='N2', save=False, method='diff')