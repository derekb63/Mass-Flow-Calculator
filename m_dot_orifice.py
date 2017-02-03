# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:20:21 2017

@author: beande
"""

import numpy as np
import cantera as ct


def conv_in_m(measurement_to_convert, starting_unit, ending_unit):
    if starting_unit == ending_unit:
        output = measurement_to_convert
    elif starting_unit == 'in' and ending_unit == 'm':
        output = np.multiply(measurement_to_convert, 0.0254)
    elif starting_unit == 'm' and ending_unit == 'in':
        output = np.divide(measurement_to_convert, 0.0254)
    else:
        print('Unit combination is not recognized')
    return output


# Convert from Pa to Psi and from psi to Pa
def conv_Pa_psi(value, starting_unit, ending_unit):
    if starting_unit == ending_unit:
        output = value
    elif starting_unit == 'psi' and ending_unit == 'Pa':
        output = np.multiply(value, 6894.75728)
    elif starting_unit == 'Pa' and ending_unit == 'psi':
        output = np.multiply(value, 0.000145037738007)
    else:
        print('Unit combination is not recognized')
    return output


def A_orf(D):
    A_orf = np.pi / 4 * D**2
    return A_orf


def Calc_Props(Gas, T, P):
    gas = ct.Solution('gri30.cti')
    gas.TPX = T, P, '{0}:1'.format(Gas)
    rho = gas.density
    k = gas.cp_mass/gas.cv_mass
    MW = gas.mean_molecular_weight
    return rho, k, MW

c_d = .99
D = conv_in_m(0.047, 'in', 'm')
A = A_orf(D)
P = conv_Pa_psi(100, 'psi', 'Pa')
P_d = 101325
T = 298
Gas = 'C3H8'
(rho, k, MW) = Calc_Props(Gas, T, P)
diff = 1
m_dot = 0
while diff > 1E-5:
    m_dot_1 = c_d*A*np.sqrt(k*rho*P*np.power(np.divide(2, k+1),
                                             np.divide(k+1, k-1)))
    c_d = np.divide(m_dot, A*np.sqrt(rho*2*(P-P_d)))
    diff = np.abs(m_dot - m_dot_1)
    if m_dot > m_dot_1:
        print(True)
        m_dot = m_dot_1 + diff/2
    else:
        print(False)
        m_dot = m_dot_1 - diff/2
    print(diff)
    print(c_d)
    print(m_dot_1)
