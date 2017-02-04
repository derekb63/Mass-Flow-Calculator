# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:13:50 2017

@author: beckp
"""
from functions import FindFile
'''
from massflowcalculator_ver_6 import mass_flow_calc
import numpy as np
import pandas as pd
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

csvName= FindFile('Open CSV')

Data=pd.DataFrame.from_csv(csvName)

Data=Data[Data.V1 < 5000]
Data=Data[Data.V1 > 500   ]

print(Data)

threedee = plt.figure().gca(projection='3d')
#threedee.plot_trisurf(Data['Phi'], Data['Diluent (N2)'], Data['V1'])
threedee.scatter(Data['Phi'], Data['Diluent (N2)'], Data['V1'])
threedee.set_xlabel('Phi')
threedee.set_ylabel('Diluent')
threedee.set_zlabel('V1')
plt.show()





























