# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 17:01:08 2015

@author: Ziang
"""
import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
from util.inverse import inverse
A = np.array([1,2,\
              4,5,\
              7,8]).reshape(3,2)
K = np.exp(- cdist(A, A, metric='euclidean', p=2) / (2 * 1**2))
L = sp.linalg.cholesky(K+1e-6*np.eye(3,3), lower=True)
K_inv = sp.linalg.cho_solve((L, True), np.eye(3,3))
print(K.dot(K_inv))

print(inverse(K))