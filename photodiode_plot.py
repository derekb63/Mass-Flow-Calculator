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
#    diff_val = new_data.diff()
    # Plot the desired signals
    new_data.plot('-', linewidth=3)
#    # Plot locations of the maximum values of the signals
#    plt.plot(new_data.idxmax(), new_data.max(), marker='o',
#             linestyle='None', markersize=8, label='Max', color='black',
#             markerfacecolor='black')
#    # Plot locations of the maximum values of the 1st finite difference
#    plt.plot(diff_val.idxmax(), np.diag(new_data.loc[diff_val.idxmax(),
#                                        new_data.columns]), marker='s',
#             linestyle='None', markersize=8, label='Grad', color='black')
    plt.legend()
    plt.show()
#    PDdata.plot(y=list(PDdata.columns[plot_num*4:(plot_num*4)+4]))
    return new_data


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


if __name__ == '__main__':

    # PDname = FindFile('Photodiode')
    # [pd1, pd2, pd3, pd4] = signal_plot(PDname, method='max')
    data = signal_plot(method='diff')
