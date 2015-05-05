# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:18:46 2015

@author: Ziang
"""

import numpy as np

a = np.arange(15).reshape(5, 3)
b = np.arange(9).reshape(3, 3)

d1 = a.dot(b).dot(a.T).diagonal()

d2 = np.einsum('ij,ji->i', a.dot(b), a.T)

d3 = np.einsum('ip,pq,iq->i', a, b, a)

d4 = np.einsum('ij,ij->i', a.dot(b), a)