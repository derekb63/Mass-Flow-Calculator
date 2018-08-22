# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 10:47:36 2018

@author: derek
"""
import re
import numpy as np
from nptdms import TdmsFile
from itertools import groupby
from functions import mass_flow, A_orf, Calc_Props
import pandas as pd
import cantera as ct
import scipy.signal as signal
import matplotlib.pyplot as plt


'''
this code is for the analysis of the oxygen-pde data based on the new data
storage method that stores all of the data in a single tdms file rather than
three separate files. Thsi process started with the data collected 7_11_2018.
'''

R = ct.gas_constant / 1000  # Gas constant (kPa m^3/kmol-K)

def import_data(filepath):
    """
    Import the data saved in the tdms file into a dataframe for processing
    
    Inputs:
        filepath: the location of the tdms file in a python readable string

    Outputs:
        data: The data from the tdms file organized in a Pandas dataframe
    """
    
    data_file = TdmsFile(filepath)
    
    test_data = data_file.as_dataframe()
    
    new_columns = {}
    for names in test_data.columns:
        new_name = re.sub("[/'/()]",' ',  names).replace('Untitled', '').strip()
        new_name = re.sub(r"\s+", '_', new_name)
        new_columns[names]  = new_name

    return test_data.rename(columns=new_columns)

def group_channels(data):
    def grouper(data):
        return [data.loc[:, x] for x in  list(zip(data.columns[::2],
                                                          data.columns[1::2]))]
    predet_channel_names = []
    photodiode_channel_names = []
    pde_channel_names = []
    for column in data.columns:
        if 'predet' in column.lower():
            predet_channel_names.append(column)
        if 'photo' in column.lower() or 'coil' in column.lower():
            photodiode_channel_names.append(column)
        else:
            pde_channel_names.append(column)

    return data.loc[:, predet_channel_names],\
           data.loc[:, pde_channel_names],\
           data.loc[:, photodiode_channel_names]


def get_pressure(serial_number, current_value):
    '''
        Return the calibration constants for a selected pressure transducer.
        The constants are stored by the serial number 
    '''
    pressure_cal_curves ={'7122122':    [34537.50, -70360.998],
                          '7122121':    [34455.79, -66451.671],
                          '071015D091': [214971.2489, -86260.085],
                          '071015D085': [215260.828 , -864671.512],
                          '1731910192': [215322880.9, -863637.3],
                          '1731910208': [215322880.9, -863637.3],
                          '1731910205': [215322880.9, -863637.3]
                          }
    pressure_value = pressure_cal_curves[serial_number][0]*current_value +\
                     pressure_cal_curves[serial_number][1]
    return pressure_value


def column_grouper(column_list):
    '''
     Take in a list of column headers, remove the time columns and organize the
     list by test and return a list of lists where each sublist contains all of
     the column names for each test. 

        Inputs:
            column_list: a list of stings that contain the names of the
                         column titles for the pandas dataframe that is being
                         analyised
        Outputs:
            a list of lists that contains the groups of column names organized
            by the first few letters
    '''
    return [list(g) for k,
             g in groupby([x for x in column_list if 'time'
                           not in x.lower() and 'coil' not in x.lower()],
            key=lambda x: x[0:4])]


def butter_filter(data, n=3, wn=0.01):
    b, a = signal.butter(n, wn, output='ba', btype='lowpass')
    return signal.filtfilt(b, a, data)
    


def velocity_calculation(photodiode_data):
    '''
        Calculate the velocity of the detonation wave from the photodiode
        signals
        
        Inputs:
            photodiode_data: the dataframe with the raw photodiode_data in it
        
        Outputs:
            velocities: a list containing a tuple of arrays that contains the
                        velocity data. The list elements are data for each fire
                        and the array elements correspond to the two velocities
                        from the photodiode signals. The first element of the
                        tuple contains the max gradient method of velocity
                        caclulation and the second element contains the
                        maximum value calculation
    '''
    def v_calc(signals, spacing=0.0762, sample_frequency=1e-6):
        '''
            Calculate the velocity based on single test signals
            
            Inputs:
                signals: the pandas dataframe that contains the photdiode data
                         for one test
                spacing: the distance between the photodiodes
                sample_frequency: the data sampling frequency
        '''
        # Determine the number of points between the maximum value and 
        # the maximum gradient
        difference_diff = np.diff(signals.diff().idxmax().values)
        difference_max = np.diff(signals.idxmax().values)

        # caclulate the valocity using the input parameters
        return (spacing/(sample_frequency*difference_diff),
               spacing/(sample_frequency*difference_max))
    # map the test data to the v_calc function to get the velocities
    return list(map(v_calc,
                    [photodiode_data.loc[:, x].apply(butter_filter, axis=0) for
                     x in column_grouper(photodiode_data.columns)]))
                 

def flow_temp_press(flow_data, ox_ducer_serial, fuel_ducer_serial,
                            ox_species='O2', fuel_species='CH4'):
    '''
        Format the data collected for the oxygen PDE in a manner that can
        utilize the existing functions developed for the dilution PDE
        
        Inputs:
            flow_data: A pandas dataframe that contains the pressure and
                        temperature data for the tests
        Outputs:
            avg_values: a dictionary with a key for each test number that
                        contains the average temperature and pressure for
                        the fuel and oxidizer for each test 
    '''
    flow_sorting = [sorted(x) for x in column_grouper(flow_data)]
    flow_sorting = [(x[0:2], x[2:]) for x in flow_sorting]
    avg_values = {}
    idx = 0
    for fuel, ox  in flow_sorting:
        avg_values[idx] = {fuel_species:
                          {'pressure': get_pressure(fuel_ducer_serial,
                                                   flow_data.loc[:, fuel].mean().values[0]),
                           'temp':flow_data.loc[:, fuel].mean().values[1]},
                           ox_species:
                          {'pressure': get_pressure(ox_ducer_serial, 
                                                    flow_data.loc[:, ox].mean().values[0]),
                           'temp': flow_data.loc[:, ox].mean().values[1]}}
        idx += 1
    return avg_values


def flow_properties(property_data, fuel_orifice_diameter, ox_orifice_diameter):
    A_fuel_orf = A_orf(fuel_orifice_diameter)
    A_ox_orifice = A_orf(ox_orifice_diameter)
    rho, k, MW = calc_props(gas, T, P)
    for key in property_data.keys():
        pass
    return data


if __name__ == '__main__':
    filename = 'C:/Users/derek/Desktop/8_21_2018/test.tdms'
    velocity_data = [None]*19
#    data = import_data(filename)
    
#    for i in range(19):
#        if i == 0:
#            velocity_data[i] = data.loc[:, 'Test_  Voltage_0':'Test_  Voltage_2']
#        else:    
#            velocity_data[i] = data.loc[:, 'Test_{0}   Voltage_0'.format(i):'Test_{0}   Voltage_2'.format(i)]
    try:
        type(predet_data)
    except NameError:
        predet_data, pde_data, photo_data = group_channels(import_data(filename))

    data = velocity_calculation(photo_data)
#    filter_example = butter_filter(photo_data[column_grouper(photo_data)[0][0]].values)
#    plt.plot(filter_example)
#    plt.plot(photo_data[column_grouper(photo_data)[0][0]].values)
#    predet_press_temp = flow_temp_press(predet_data,
#                                               '1731910192',
#                                               '1731910208')
    #del photo_data
    
    # TODO: The column grouper and group channels functions are pretty much redundant