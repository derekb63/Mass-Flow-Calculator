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
def signal_plot(PDname, method='max'):
    PDfile = TdmsFile(PDname)
    PDdata = PDfile.as_dataframe(time_index=True, absolute_time=False)
#    PD1 = PDdata[PDdata.columns[0::4]]
#    PD2 = PDdata[PDdata.columns[1::4]]
#    PD3 = PDdata[PDdata.columns[2::4]]
#    PD4 = PDdata[PDdata.columns[3::4]]
#    return(PD1, PD2, PD3, PD4)
    num_tests = len(PDdata.columns)/4
    plot_num = test_enter(num_tests)

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
        return plot_num

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

    PDname = FindFile('Photodiode')
#    [pd1, pd2, pd3, pd4] = signal_plot(PDname, method='max')
    data = signal_plot(PDname, method='max')
