#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 12:17:45 2016

@author: beande
"""
import numpy as np
import pandas as pd
import cantera as ct
from nptdms import TdmsFile
from tkinter import *
from tkinter.filedialog import askopenfilename
import sys


R = ct.gas_constant / 1000                     # Gas constant (kPa m^3/kmol-K)
P_d = 101325                                  # Downstream pressure in kPa


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


def calibrate(P1, T1, ducer, cals, Gas):
    gas = ct.Solution('gri30.xml')
    cor_p = P1*cals[ducer-1][0]+cals[ducer-1][1]+14.7
    Pres = cor_p*6894.75729                   # Orifice Upstream Pressure
    off = int(np.argmax(-np.diff(Pres)))        # Valve turns on

    P_u = Pres[Pres.index[off-75]:Pres.index[off-1]]
    P_u = np.mean(P_u)

    Ttime = pd.Series(data=np.abs(T1.index-Pres.index[off-75]), index=T1.index)
    # Orifice Upstream Temperature
    T = T1[Ttime.idxmin():Ttime.index[Ttime.index.get_loc(Ttime.idxmin())+10]]

    T_avg = np.mean(T)

    # Get Properties Based on P and T

    gas.TPX = T_avg, P_u, '{0}:1'.format(Gas)
    rho = gas.density
    k = gas.cp_mass/gas.cv_mass
    MW = gas.mean_molecular_weight

    return P_u, T_avg, rho, k, MW


def mass_flow(k, R, MW, rho, A_orifice, A_tube, P_u, P_d, T_avg, C_d=0.99):
    # finds mass flow rate
    # two different equations depends on sub/supersonic
    if P_u/P_d >= ((k+1)/2)**((k)/(k-1)):
        m_dot = A_orifice * P_u * k * C_d * \
            ((2/(k+1))**((k+1)/(k-1)))**(0.5)\
            / ((k*(R/MW)*T_avg))**(0.5)
    else:

        m_dot = rho * A_orifice * C_d \
                * ((2*(P_u-P_d))/(rho*(1-(A_orifice/A_tube)**2)))**(0.5)
    return m_dot


def reformat(data):
    index = list(data.index)     # time
    labels = list(data.columns)  # test_num/channel_num

    # finds the number of channels recorded in the test
    # basically looks for the max number in labels, channel_num
    numChannels = []
    for el in labels:
        x = el.split('/')
        x.pop(0)
        x.pop(0)
        x = x[0]
        if x[-2].isdigit():
            mynumber = x[-2]
        if x[-1].isdigit():
            mynumber += x[-1]
        numChannels.append(int(mynumber))
    numChannels = max(numChannels)+1

    mybiglist = []
    for _ in range(int(len(labels)/numChannels)):
        mybiglist.append(pd.DataFrame(index=index, columns=[]))

    for name in labels:
        L = name.split("/")

        try:
            test_num = int(L[1][9:-1])
        except ValueError:
            test_num = 0
        PTnum = ''
        for el in L[2]:
            try:
                PTnum = int(el)+1
            except:
                pass
        try:
            int(PTnum)
        except:
            PTnum = 0
        mybiglist[test_num]['Gauge'+str(PTnum)] = data[name]

    return mybiglist


def find_M_dot(Tempdata, Pressdata, test, ducer, TC, D_orifice, cals, Gas):
    # Tempdata is a list of dataframes
    # Pressdata is a list of dataframes
    # test is an int representing the test number
    # ducer is the PT number that is used to calculate the mass flow rate
    # Current_0 correlates to ducer=1
    # TC is the number of the thermocouple used to calculate the mass flow rate
    # Temperature_0 correlates to TC=1
    # D_orifice is in inches
    # Gets a Series out of the list of dataframes
    Tdata = Tempdata[test]
    Pdata = Pressdata[test]
    # print(Pdata)
    P1 = Pdata['Gauge'+str(ducer)]
    T1 = Tdata['Gauge'+str(TC)]
    # *C to K
    if np.mean(T1) < 200:
        T1 = T1+273

    # Calculate the flow area of the sonic orifice
    A_orifice = (np.pi/4)*(D_orifice*0.0254)**2  # Area of the orifice in m
    D_tube = 0.018  # Tube ID in INCHES
    A_tube = (np.pi/4)*(D_tube*0.0254)**2  # Area of the orifice in m

    # Check to see if the pressure transducer data makes sense
    # Apply the Calibrations to the pressure transducer data
    # print(P1)
    [P_u, T_avg, rho, k_gas, MW] = calibrate(P1, T1, ducer, cals, Gas)

    # Mass Flow Calculation
    m_dot = mass_flow(k_gas, R, MW, rho, A_orifice, A_tube, P_u, P_d, T_avg)

    return m_dot


def velocity_calc(PDname, method='max'):
    PDfile = TdmsFile(PDname)
    PDdata = PDfile.as_dataframe(time_index=True, absolute_time=False)
    # PDdata.to_pickle('IP.pkl')
    PD1 = PDdata[PDdata.columns[0::4]]
    # print(PD1)
    PD2 = PDdata[PDdata.columns[1::4]]
    PD3 = PDdata[PDdata.columns[2::4]]
    PD4 = PDdata[PDdata.columns[3::4]]
#    print(PDdata.columns)
#    PDdata[PDdata.columns[-4:-1]].plot()
#    PD1.plot()
#    PD2.plot()
#    PD3.plot()
#    PD4.plot()
    del PDdata
    # Choose the method for the determination of the velocity
    if method == 'diff':
        D1 = PD1.diff()
        D2 = PD2.diff()
        D3 = PD3.diff()
        D4 = PD4.diff()
    elif method == 'max':
        D1 = PD1
        D2 = PD2
        D3 = PD3
        D4 = PD4
    else:
        sys.exit('The method you have chosen for the velicty calculation is' +
                 ' not reconized. Please select a different method and retry.')

    del PD1, PD2, PD3, PD4
    t1 = D1.idxmax()
    t2 = D2.idxmax()
    t3 = D3.idxmax()
    t4 = D4.idxmax()
    del D1, D2, D3, D4
    L1 = 0.127762
    L2 = 0.129337
    L3 = 0.130175

    T1 = pd.Series(t2.values - t1.values)
    T2 = pd.Series(t3.values - t2.values)
    T3 = pd.Series(t4.values - t3.values)
    V1 = L1/T1.values
    V2 = L2/T2.values
    V3 = L3/T3.values
    R1 = np.sqrt((-.5*(L1/T1.values**2)*1e-6)**2+(1/T1.values*0.003175)**2)
    R2 = np.sqrt((-.5*(L2/T2.values**2)*1e-6)**2+(1/T2.values*0.003175)**2)
    R3 = np.sqrt((-.5*(L3/T3.values**2)*1e-6)**2+(1/T3.values*0.003175)**2)

    vel_data = pd.DataFrame(np.transpose(
            np.vstack((V1, V2, V3, R1, R2, R3))))
    vel_data.columns = ['V1', 'V2', 'V3', 'R1', 'R2', 'R3']

    # del T1, T2, T3, V1, V2, V3, R1, R2, R3, L1, L2, L3, t1, t2, t3, t4
    # print (vel_data)
    # plt.plot(phi, vel_data[:,0], 'x')
    return vel_data


# Fuel_Oxidizer_Ratio takes the imput strings of the fuel and oxidizer as well
# and caclulates the stoiciometric fuel oxidizer ratio
def Fuel_Oxidizer_Ratio(fuel='C3H8', ox='N2O'):
    Elements = ['C', 'H', 'O', 'N']
    ox_val = [0, 0, 0, 0]
    if ox == 'Air':
        ox = ['O2', 'N2']
        ox_val = [0, 0, 2, 7.52]
        MW_ox = 28.8
    elif ox == 'N2O':
        ox_val = [0, 0, 1, 2]
        MW_ox = 44
    elif ox == 'O2':
        ox_val = [0, 0, 2, 0]
        MW_ox = 32
    else:
        print('Your Oxidizer is not reconized')
    if fuel == 'H2':
        fuel_val = [0, 2, 0, 0]
        MW_fuel = 2
    elif fuel == 'CH4':
        fuel_val = [1, 4, 0, 0]
        MW_fuel = 16.04
    elif fuel == 'C3H8':
        fuel_val = [3, 8, 0, 0]
        MW_fuel = 44
    else:
        print('Your Fuel is not reconized')
    react_names = [fuel]
    react_names += [ox]
    product_vals = [(1, 0, 2, 0), (0, 2, 1, 0), (0, 0, 0, 2)]
    product_names = ['CO2', 'H2O', 'N2']
    names = [ox] + product_names
    A = pd.DataFrame(np.transpose(np.vstack([ox_val, product_vals])),
                     index=Elements, columns=names)
    coeffs = np.abs(np.linalg.solve(A[:][:], [-x for x in fuel_val]))
    F_O_s = (1*MW_fuel)/(coeffs[0]*MW_ox)
    return F_O_s


    