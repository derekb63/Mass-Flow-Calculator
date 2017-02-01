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


# Read in the photodiode data and plot the point that is used for the
# measurement method defined in the function call
def signal_plot(PDname=None, method='max'):
    if PDname is None:
        PDname = FindFile('Photodiode')
    PDfile = TdmsFile(PDname)
    PDdata = PDfile.as_dataframe(time_index=True, absolute_time=False)
#    PD1 = PDdata[PDdata.columns[0::4]]
#    PD2 = PDdata[PDdata.columns[1::4]]
#    PD3 = PDdata[PDdata.columns[2::4]]
#    PD4 = PDdata[PDdata.columns[3::4]]
#    return(PD1, PD2, PD3, PD4)
    num_tests = int(len(PDdata.columns)/4)
    plot_num = test_enter(num_tests)
    col_names = []
    for i in range(num_tests):
        col_names += ['Test {0} '.format(i) + 'PD1',
                      'Test {0} '.format(i) + 'PD2',
                      'Test {0} '.format(i) + 'PD3',
                      'Test {0} '.format(i) + 'PD4']
    PDdata.columns = col_names
    PDdata.plot(y=list(PDdata.columns[plot_num*4:(plot_num*4)+4]))
    return PDdata


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

def FindFile(text):
    def openFile():
        global Fname
        Fname = askopenfilename()
        print(Fname)
        root.destroy()

    root = Tk()
    root.attributes("-topmost", True)
    Button(root, text=text, command = openFile).pack(fill=X)
    mainloop()     

    return Fname

if __name__ == '__main__':

    # PDname = FindFile('Photodiode')
#    [pd1, pd2, pd3, pd4] = signal_plot(PDname, method='max')
    signal_plot(method='max')
