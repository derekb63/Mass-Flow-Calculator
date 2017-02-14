#! usr/bin/env python
from massflowcalculator import mass_flow_calc as m
Tname = 'D:\PDE Project\Dilution Project\Dilution Experiment Tests\Phase 1\January 31\TC2.tdms'
Pname = 'D:\PDE Project\Dilution Project\Dilution Experiment Tests\Phase 1\January 31\PT2.tdms'
PDname = 'D:\PDE Project\Dilution Project\Dilution Experiment Tests\Phase 1\January 31\PD2.tdms'
Diluent = 'N2'
Data    = m(diluent=Diluent, Tname=Tname, Pname=Pname, PDname=PDname, save='True', method='diff')