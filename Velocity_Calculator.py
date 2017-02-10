# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 15:35:21 2017

@author: beckp
"""
'''
Choose a folder specified by filepath
and it runs the mass flow calculator for each 
tdms file in there and creates a csv of the data

'''
import pandas as pd
from massflowcalculator import mass_flow_calc
import glob, os 
#Change this variable based on which folder you want to run
filepath='D:\PDE Project\Dilution Project\Dilution Experiment Tests\Phase 1\January 31\CO2'

#changes working directory of script
os.chdir(filepath)

#finds all tdms files in folder
#puts filenames into a list
#I am not sure if it looks for tdms files within 
#other folders inside of specified folder

files=glob.glob("*.tdms")
files.sort()
filesets=[]
print(files)
'''
-------------IMPORTANT NAMING CONVENTION-------------
This needs three files to work. 

-Photodiode filenames need to start with 'PD'
-Pressure Transducer filenames need to start with 'PT'
-Thermocouple filenames need to start with 'TC'

-All three filenames, after their respective starts, 
need to have the same endings.
-Traditionally it has been an integer, 
however it does not need to be

E.g. 'PD1','PT1','TC1' will all be grouped together
'PD2','TC3','PT4' will not be grouped together
------------------------------------------------------
'''
#finds the three files that are associated with each other 
for file in files:
    if file[0:2]=='PD':
        testbatch=[file]
        for el in files:
            if el[2:-5]==file[2:-5]:
                if el != file:
                    testbatch.append(el)
        filesets.append(testbatch)
        
        
        
#creates blank Dataframe
Data = pd.DataFrame(columns=['Phi', 'Diluent (N2)', 'V1', 'V2', 'V3', 'R1', 'R2', 'R3'])

#files the PD, PT,TC filenames and runs mass_flow_calc
#which outputs a Dataframe of the same style (same columns)
#as the one above
#appends the new Dataframe to the existing one
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
     

