#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 19:32:20 2016

@author: beande
"""
import numpy as np
import pandas as pd
from nptdms import TdmsFile
from functions import reformat, find_M_dot, velocity_calc, FindFile
from functions import Fuel_Oxidizer_Ratio


'''
This code is the main fuction for the massflowcalculator that brings all of the 
functions together. To run the code call the mass_flow_calc function with the 
required inputs. Only change things in the if __name__ == '__main__' block

Inputs:
    fuel: a string readable by cantera that denotes the fuel species
    oxidizer: a cantera readable string that denotes the oxidizer species
    diluent: a cantera readable string that denotes the oxidizer species
    Tname: a filepath sting that points to the tdms file that contains the 
           temperature data from the experiment
    Pname:  a filepath sting that points to the tdms file that contains the 
           pressure data from the experiment
    PDname:  a filepath sting that points to the tdms file that contains the 
           photodiode data from the experiment
    save: boolean that determines if the output data is saved as a csv in the
          PDname directory
    method: max or diff strings to state which method to use for the
            velocity determination
    dil_orifice: diameter of the orifice used for metering the diluent
                 during the tests

Outputs:
    Data: the caluclated data values for the input data. the data contains the
          equivalence ratio, dilution mass fraction, velocity and the velocity
          error. If the fucntion is passed a true value for the save varaible
          the Data dataframe will be saved as a csv.
'''
def mass_flow_calc(fuel='C3H8', oxidizer='N2O', diluent=None,
                   Tname=None, Pname=None, PDname=None, save=True,
                   method='max', dil_orifice=0.063):
    start = time.time()


    Gases = [oxidizer, fuel, diluent]  # Species of the gas used ct form

    """ If you don't want to choose the files """
    # Tname  = 'D:\PDE Project\Dilution Project\Dilution Experiment Tests\Phase 1\February 3\Dilution_none\TC2.tdms'
    # Pname  = 'D:\PDE Project\Dilution Project\Dilution Experiment Tests\Phase 1\February 3\Dilution_none\PT2.tdms'
    # PDname = 'D:\PDE Project\Dilution Project\Dilution Experiment Tests\Phase 1\February 3\Dilution_none\PD2.tdms'

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
    # reformats the Dataframes into usable structure (list of Dataframes)
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

    # Finds mass flow rate for each gas on each test
    for Gas in Gases:
        # For each gas, defines which PT and TC it is using,
        # as well as the orifice size we are using
        if (Gas == 'Propane') or (Gas == 'C3H8'):
            ducer = 6
            TC = 1
            D_orifice = 0.047  # Diameter of the orifice in INCHES
        elif (Gas == 'NitrousOxide') or (Gas == 'N2O'):
            ducer = 5  # On PDE ducer = 5 changed to 6 for testing
            D_orifice = 0.142  # Diameter of the orifice in INCHES
            TC = 2
        elif (Gas == 'Nitrogen') or (Gas == 'N2'):
            ducer = 8
            D_orifice = dil_orifice  # Diameter of the orifice in INCHES
            TC = 3
        elif (Gas == 'CO2') or (Gas == 'CarbonDioxide'):
            ducer = 8
            D_orifice = dil_orifice  # Diameter of the orifice in INCHES
            TC = 3
        else:
            print('Gas Not Recognized')
        # Finds mass flow rate for each test
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
    for test in range(len(dilution)):
        if abs(dilution[test]) < 1e-4:
            dilution[test] = 0.0

    # Place the equivalence ratio and dilution data into a pandas.DataFrame
    # so that it can be easily managed/saved later

    Data = pd.concat([phi, dilution], axis=1)

    del Pressdata, Tempdata, Pressfile, Tempfile, M_dot, m_dot,  # dilution,phi

    # Put the velocity and error data into the Data pandas.DataFrame
    # The necessary data is now all in the same place

    Data = pd.concat([Data, velocity_calc(PDname, method)], axis=1)

    Data.index.name = 'Test Number'

    # Write the data file to the same location as the photodiode data was
    # sourced by creating a new file or appending to the file
    if save is True:
        Save_File = '/'.join(PDname.split('/')[:-1]) + '/' + 'testdata.csv'
        Data.to_csv(Save_File, mode='a')
        print('The data has been saved to {0}'.format(Save_File))
    # print('Run Time:', end-start, 'seconds')
    return Data

if __name__ == '__main__':
	# This is where all of the inputs go to do the analysis
    filepath = 'D:\PDE Project\Dilution Project\Dilution Experiment Tests\Phase 1\February 7\eighth_in_orifice\Itrogen\psi0'
    Data = mass_flow_calc(diluent='N2',
                          Tname=filepath + '\TC.tdms',
                          Pname=filepath + '\PT.tdms',
                          PDname=filepath + '\PD.tdms',
                          save=False,
						  method='diff')

    # Plots this data so we can see what kind of curve we are getting
    Data.plot(x='Phi', y=['V1'], marker='x', linestyle='None',
              ylim=(500, 3500), xlim=[0.95, 1.05])
