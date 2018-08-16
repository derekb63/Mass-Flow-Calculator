# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 10:47:36 2018

@author: derek
"""
import os, re, sys
import numpy as np
import pandas as pd
from nptdms import TdmsFile
from functions import reformat, find_M_dot, velocity_calc, FindFile
from functions import Fuel_Oxidizer_Ratio
from itertools import groupby
from multiprocessing import Pool, TimeoutError

'''
this code is for the analysis of the oxygen-pde data based on the new data
storage method that stores all of the data in a single tdms file rather than
three separate files. Thsi process started with the data collected 7_11_2018.
'''


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

    return test_data .rename(columns=new_columns)

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


def get_pressure_cal(serial_number):
    '''
        Return the calibration constants for a selected pressure transducer.
        The constants are stored by the serial number 
    '''
    pressure_cal_curves ={'7122122':    [34537.50, -70360.998],
                          '7122121':    [34455.79, -66451.671],
                          '071015D091': [214971.2489, -86260.085],
                          '071015D085': [215260.828 , -864671.512],
                          '1731910192': [215322880.9, 863637.3],
                          '1731910208': [215322880.9, 863637.3],
                          '1731910205': [215322880.9, 863637.3]
                          }
    return pressure_cal_curves[serial_number]


def velocity_calculation(photodiode_data):
    
    def v_calc(signals, spacing=0.0762, sample_spacing=1e-6):
        difference_diff = np.diff(signals.diff().idxmax().values)
        difference_max = np.diff(signals.idxmax().values)
        return (spacing/(sample_spacing*difference_diff),
                spacing/(sample_spacing*difference_max))
    
    photodiode_data = photodiode_data.loc[:,
                                           [x for x in photodiode_data.columns
                                            if 'time' not in x.lower()
                                            and 'coil' not in x.lower()]]
    grouped_columns = [list(g) for k, g in groupby(photodiode_data.columns,
                       key=lambda x: x[0:2])]
    grouped_data = [photodiode_data.loc[:, x] for x in grouped_columns]
    data = map(v_calc, grouped_data)
    
    return list(data)


if __name__ == '__main__':
    filename = 'C:/Users/derek/Desktop/8_15_2018/test.tdms'
    velocity_data = [None]*19
#    data = import_data(filename)
    
#    for i in range(19):
#        if i == 0:
#            velocity_data[i] = data.loc[:, 'Test_  Voltage_0':'Test_  Voltage_2']
#        else:    
#            velocity_data[i] = data.loc[:, 'Test_{0}   Voltage_0'.format(i):'Test_{0}   Voltage_2'.format(i)]
    _, _, photo_data = group_channels(import_data(filename))
    data = velocity_calculation(photo_data)

    # rename the pressure transducer channels with the pressure tranducer 
    # serial number
    