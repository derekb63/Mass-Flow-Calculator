# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:20:21 2017

@author: beande
"""

import numpy as np
from functions import conv_in_m, conv_Pa_psi, A_orf, Calc_Props

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
