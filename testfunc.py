# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import cantera as ct
from nptdms import TdmsFile
from tkinter import *
from tkinter.filedialog import askopenfilename



def reformat(data):
    index = list(data.index)#time
    labels = list(data.columns)#test_num/channel_num

    #finds the number of channels recorded in the test
    #basically looks for the max number in labels, channel_num
    numChannels=[]
    for el in labels:
        x=el.split('/')
        x.pop(0)
        x.pop(0)
        x=x[0]
        if x[-2].isdigit():
            mynumber=x[-2]
        if x[-1].isdigit():
            mynumber+=x[-1]
        numChannels.append(int(mynumber))
    numChannels=max(numChannels)+1
    
    mybiglist = []
    print(len(labels))
    print(numChannels)
    print(range(int(len(labels)/numChannels)))
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
      
        
if __name__ == '__main__':
    reformat(Tempdata)
