# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 15:35:21 2017

@author: beckp
"""

import pandas as pd
from massflowcalculator import mass_flow_calc
import glob, os 

filepath='D:\PDE Project\Dilution Project\Dilution Experiment Tests\Phase 1\January 31\CO2'
os.chdir(filepath)

files=glob.glob("*.tdms")
files.sort()
filesets=[]
print(files)
for file in files:

    if file[0:2]=='PD':
        testbatch=[file]
        for el in files:
            if el[2:-5]==file[2:-5]:
                if el != file:
                    testbatch.append(el)
        filesets.append(testbatch)
Data = pd.DataFrame(columns=['Phi', 'Diluent (N2)', 'V1', 'V2', 'V3', 'R1', 'R2', 'R3'])
for el in filesets:
    for name in el:
        if name[0:2]=='PD':
            PDFile=name
            print(name)
        if name[0:2]=='PT':
            PTFile=name
        if name[0:2]=='TC':
            TCFile=name
            
    newData=mass_flow_calc(fuel='C3H8', oxidizer='N2O', diluent='N2',
                   Tname=TCFile, Pname=PTFile, PDname=PDFile, save=False,
                   method='diff')
    Data=Data.append(newData)
csvName=filepath+ '/' + 'masterfile.csv'

Data.to_csv(csvName)
     

