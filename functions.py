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


R = ct.gas_constant / 1000                     # Gas constant (kPa m^3/kmol-K)
P_d = 101325                                  # Downstream pressure in kPa

gas = ct.Solution('gri30.cti')


def calibrate(P1, T1, ducer, cals, Gas):
    cor_p = P1*cals[ducer][0]+cals[ducer][1]+14.7
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


def mass_flow(k, R, MW, rho, A_orifice, A_tube, P_u, P_d, T_avg):
    # print(P_u,rho,k,T)
    if P_u/P_d >= ((k+1)/2)**((k)/(k-1)):
        m_dot = A_orifice * P_u * k * \
            ((2/(k+1))**((k+1)/(k-1)))**(0.5)\
            / ((k*(R/MW)*T_avg))**(0.5)
    else:

        m_dot = rho*A_orifice\
                * ((2*(P_u-P_d))/(rho*(1-(A_orifice/A_tube)**2)))**(0.5)
    return m_dot


def reformat(data, numChannels):
    index = list(data.index)
    labels = list(data.columns)

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
    '''
    if np.mean(P1)<0.0035:
    print('This pressure transducer is probably not connected')
    elif np.mean(P1)>0.020:
    print('This pressure transducer is probably broken')
    else:
    '''
    # Apply the Calibrations to the pressure transducer data
    # print(P1)
    [P_u, T_avg, rho, k_gas, MW] = calibrate(P1, T1, ducer, cals, Gas)

    # Mass Flow Calculation
    m_dot = mass_flow(k_gas, R, MW, rho, A_orifice, A_tube, P_u, P_d, T_avg)
    '''
    print('k= '+str(k_gas))
    print('rho= '+str(rho))
    print(A_orifice)
    print(A_tube)
    print('P_u= '+str(P_u))
    print('P_d= '+str(P_d))
    print('T= '+str(T_avg))
    print(Gas,test)
    print(m_dot)
    print()
    '''
    return m_dot
'''
if __name__ == '__main__':
    Pressdata=Pressfile.as_dataframe(time_index=True,absolute_time=False)
    L=reformat(Pressdata,8)
'''


def velocity_calc(PDname):
    PDfile = TdmsFile(PDname)
    PDdata = PDfile.as_dataframe(time_index=True, absolute_time=False)
    # PDdata.to_pickle('IP.pkl')
    PD1 = PDdata[PDdata.columns[0::4]]
    # print(PD1)
    PD2 = PDdata[PDdata.columns[1::4]]
    PD3 = PDdata[PDdata.columns[2::4]]
    PD4 = PDdata[PDdata.columns[3::4]]
    del PDdata
    D1 = PD1.diff()
    D2 = PD2.diff()
    D3 = PD3.diff()
    D4 = PD4.diff()
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

    vel_data = np.transpose(np.vstack((V1, V2, V3, R1, R2, R3)))
    # del T1, T2, T3, V1, V2, V3, R1, R2, R3, L1, L2, L3, t1, t2, t3, t4
    # print (vel_data)
    # plt.plot(phi, vel_data[:,0], 'x')
    return vel_data