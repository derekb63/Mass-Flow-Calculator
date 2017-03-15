#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:32:40 2017

@author: aero-10
"""
import pandas as pd
import numpy as np
import scipy.stats.stats as sci_stats
import matplotlib.pyplot as plt
from tkinter import Button, mainloop, X, Tk
from tkinter.filedialog import askopenfilename

# Class version of the csv_reader function used for the
# combustion institute conference paper


class ProcessData:
    def findfile(self, button_text):
        def openFile():
            global Fname
            Fname = askopenfilename()
            print(Fname)
            root.destroy()
        root = Tk()
        root.attributes("-topmost", True)
        Button(root, text=button_text, command=openFile).pack(fill=X)
        mainloop()
        return Fname

    def __init__(self, file_name=None):
        try:
            # Read the data from the csv into a Pandas DataFrame
            self.file_name = str(file_name)
            self.data = pd.read_csv(self.file_name)
        except:
            self.file_name = self.findfile('csv file to plot')
            # Read the data from the csv into a Pandas DataFrame
            self.data = pd.read_csv(self.file_name)

    def stripcols(self, desired_columns=['Diluent', 'V1', 'R1']):
        self.keep_columns = []
        for i in desired_columns:
            [self.keep_columns.append(loc) for loc,
             idx in enumerate(self.data.columns) if i in idx]
        return self.data.take(self.keep_columns, axis=1)
    
    def columns(self):
        return self.data.columns
    
    def separate(self):
        self.base_data = self.data[self.data.columns[0::3]]
        self.base_data.replace(self.base_data[self.base_data.columns[0]])
        self.CO2_data = self.data[data.columns[1::3]]
        self.N2_data = self.data[self.data.columns[2::3]]
        return self.base_data, self.CO2_data, self.N2_data

    def trimdata(self, trim_limits=(500, 3000), trim_columns='V1'):
        trim_list = [i for i in self.data.columns if trim_columns in i]
        
#        self.base_data = self.base_data[self.base_data['V1'] >
#                                        min(trim_limits)]
#        self.base_data = self.base_data[self.base_data['V1'] <
#                                        max(trim_limits)]
#
#        self.CO2_data = self.CO2_data[self.CO2_data['V1.1'] > min(trim_limits)]
#        self.CO2_data = self.CO2_data[self.CO2_data['V1.1'] < max(trim_limits)]
#
#        self.N2_data = self.N2_data[self.N2_data['V1.2'] > min(trim_limits)]
#        self.N2_data = self.N2_data[self.N2_data['V1.2'] < max(trim_limits)]
    

if __name__ == '__main__':
    Fname = '/home/aero-10/Documents/Mass-Flow-Calculator/Compiled test data.csv'
    data = ProcessData(file_name=Fname)
