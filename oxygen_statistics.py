# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 15:50:47 2018

@author: derek
"""

import pickle
import glob
import matplotlib.pyplot as plt
import numpy as np

def loadPkl(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

if __name__ == "__main__":
    directory = 'C:\\Users\\derek\\Desktop\\8_28_2018\\'
    files = glob.glob(directory + '*.pkl')
    data = list()
    for f in files:
        data.append(loadPkl(f))
    v_data = np.ndarray((sum([len(x.keys()) for x in data]), 2))
    idx = 0
    for value in data:
        for key, item in value.items():
            print(item['equivalence_ratio'], item['velocity'][0].mean())
            v_data[idx, 0] = item['equivalence_ratio']
            v_data[idx, 1] = item['velocity'][0].mean()
            idx += 1
#    plt.plot(v_data[:, 0], v_data[:, 1])