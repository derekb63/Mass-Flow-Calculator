# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:13:50 2017

@author: beckp
"""
'''
This is meant to be run after a single use of either 
massflowcalculator.py or Velocity_Calculator.py

It takes the csv outputted by these functions and plots 
velocity as a function of phi and dilution ratio
'''

import numpy as np
import pandas as pd
from functions import FindFile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#tkinter gui to search for csv file. 
csvName= FindFile('Open CSV')

#imports data from csv into a pandas dataframe
Data=pd.DataFrame.from_csv(csvName)
#eliminates extraneous data points (velocities too high or low (misfires))

Data=Data[Data.V1 < 5000]
Data=Data[Data.V1 > 500   ]

#print used for debugging
#print(Data)



#most of this stuff is copied from
#https://pythonprogramming.net/matplotlib-3d-scatterplot-tutorial/
threedee = plt.figure().gca(projection='3d')

#Can either plot scatter plots or surfaces
threedee.scatter(Data['Phi'], Data['Diluent (N2)'], Data['V1'])
#threedee.plot_trisurf(Data['Phi'], Data['Diluent (N2)'], Data['V1'])

#labelling and stuff
threedee.set_xlabel('Phi')
threedee.set_ylabel('Diluent')
threedee.set_zlabel('V1')

#display plot created
plt.show()





























