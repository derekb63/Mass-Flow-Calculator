# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 15:50:47 2018

@author: derek
"""

import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

def loadPkl(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

if __name__ == "__main__":
    directory = ['C:\\Users\\derek\\Desktop\\8_21_2018\\',
                 'C:\\Users\\derek\\Desktop\\8_28_2018\\',
                 'C:\\Users\\derek\\Desktop\\8_28_2018\\']
    files = itertools.chain(*[glob.glob(x + '*.json') for x in directory])
    
    
    data = list(map(loadPkl, files))
    phis = []
    vels = []
    for group in data:
        for key, item in group.items():
            vel = np.mean(item['velocity'][0])
            if vel > 100 and  vel < 3000:
                phis.append(item['equivalence_ratio'])
                vels.append(vel)
            else: 
                pass
    
    fig, ax = plt.subplots()
    ax.plot(phis, vels, 'ob')
    ax.set(ylabel='Velocity (m/s)', xlabel='$\Phi$')
    fig.show()
            