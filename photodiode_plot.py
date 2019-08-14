#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 18:57:47 2017

@author: beande
"""
from nptdms import TdmsFile
from tkinter import *
from tkinter.filedialog import askopenfilename
import sys
import matplotlib.pyplot as plt 
import numpy as np
from functions import FindFile
from scipy import signal as s

# Read in the photodiode data and plot the point that is used for the
# measurement method defined in the function call
def signal_plot(PDname=None, method='diff'):
    if PDname is None:
        PDname = FindFile('Photodiode')
    PDfile = TdmsFile(PDname)
    PDdata = PDfile.as_dataframe(time_index=True, absolute_time=False)
    num_tests = int(len(PDdata.columns)/4)
    plot_num = test_enter(num_tests)
    new_data = PDdata[PDdata.columns[plot_num*4:(plot_num*4)+4]]

    new_data.columns = ['Test {0} '.format(plot_num) + 'PD1',
                        'Test {0} '.format(plot_num) + 'PD2',
                        'Test {0} '.format(plot_num) + 'PD3',
                        'Test {0} '.format(plot_num) + 'PD4']

    new_data.index.name = 'Time (s)'
    # Determine the values of the 1st finite difference
    diff_val = new_data.diff()
    # Plot the desired signals
    new_data.plot(linewidth=3)
    # Plot locations of the maximum values of the signals
    plt.plot(new_data.idxmax(), new_data.max(), marker='o',
             linestyle='None', markersize=8, label='Max', color='black',
             markerfacecolor='black')
    # Plot locations of the maximum values of the 1st finite difference
    plt.plot(diff_val.idxmax(), np.diag(new_data.loc[diff_val.idxmax(),
                                        new_data.columns]), marker='s',
             linestyle='None', markersize=8, label='Grad', color='black')
    plt.legend()
    plt.show()
#    PDdata.plot(y=list(PDdata.columns[plot_num*4:(plot_num*4)+4]))
    return new_data

def signal_plot_test(PDname=None, method='diff'):
    if PDname is None:
        PDname = FindFile('Photodiode')
    
        


def test_enter(num_tests):
    plot_num = input('Which test to plot out of {0}: '.format(num_tests))
    if int(plot_num) > num_tests:
        print('There are only {0} tests'.format(num_tests))
        plot_num = None
    elif int(plot_num) < 0:
        print('Please input a positive value')
        plot_num = None
    else:
        return int(plot_num)


def fft_plotter(data, Fs=1e6):
    Ts = 1.0/Fs
    t = np.arange(0, 1, Ts)
    n = len(data)
    T = n/Fs
    k =np.arange(n)
    frq = k/T # two sides frequency range
    frq = frq[range(int(n/2))] # one side frequency range
    
    Y = np.fft.fft(data)/n # fft computing and normalization
    Y = Y[range(int(n/2))]
    
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t,data)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    ax[1].plot(frq ,abs(Y),'r') # plotting the spectrum
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')
    plt.show()

if __name__ == '__main__':

    # PDname = FindFile('Photodiode')
    # [pd1, pd2, pd3, pd4] = signal_plot(PDname, method='max')
#    data = signal_plot(method='diff')
    PDname = 'D:\\Oxygen_Data\\08_06_2019\\test006C_08062019.tdms'
    data_file = TdmsFile(PDname)
    test_names = data_file.groups()
    pd_data = []
#    desired_names = ['photo', 'coil']
    desired_names = ['untitled']
    for test in test_names:
        channels = data_file.group_channels(test)
        ch_idx = [idx for idx, val in enumerate(channels) if
                  (any(ele in val.path.lower() for ele in desired_names) and
                  'time' not in val.path.lower())]

        pd_data.append([data_file.group_channels(test)[x] for x in ch_idx])
#    fft_plotter(pd_data[0][0].data)
#    fft_plotter(pd_data[0][1].data)
    plt.plot([x.data for x in pd_data[0]])
        
#    idx = 0; plt.plot(pd_data[idx][:].data)
#    test_groups = data_file.groups()
#    data = data_file.data.group_channels(test_groups[0])
#    data_dict = {dataset.channel: dataset for dataset in data}
