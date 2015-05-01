# -*- coding: utf-8 -*-
"""
Created on Fri May  1 04:38:10 2015

@author: Ziang
"""
import numpy as np

def npdot(x, y):
    return x.dot(y)

if __name__=='__main__':
    x = np.random.randn(10000, 10000)
    y = np.random.randn(10000, 10000)
    for i in range(100):
        npdot(x, y)
