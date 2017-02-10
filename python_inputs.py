#! usr/bin/env python
from massflowcalculator import mass_flow_calc as m
Tname = 'D:\PDE Project\PDE Data\Default Data\TC.tdms'
Pname = 'D:\PDE Project\PDE Data\Default Data\PT.tdms'
PDname = 'D:\PDE Project\PDE Data\Default Data\PD.tdms'
Diluent = 'N2'
Data    = m(diluent=Diluent, Tname=Tname, Pname=Pname, PDname=PDname, save=True, method='diff')