# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:11:02 2015

@author: Ziang
"""
from numpy import unique
from numpy import array
from numpy import genfromtxt
def load_ogpc_tr():
    raw = genfromtxt('../data/train_ogpc.csv', dtype=str, delimiter=',')
    data = array(raw[1:-1, 0:-2], dtype=float)
    target = raw[1:-1, -1]
    labels = unique(target)
    for i in range(len(labels)):
        target[target==labels[i]] = i
    target = array(target, dtype=int)
    return data, target