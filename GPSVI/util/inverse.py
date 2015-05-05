# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 21:16:01 2015

@author: Ziang
"""
from numpy import eye
from scipy.linalg import cholesky, cho_solve#, eigvals
#from numpy import any
def cho_inverse(K, sigma=None):
#    if any(eigvals(K) <= 0):
#        print('K is not positive definite!')
    
    if sigma is None:
        sigma = 1.0 / K.shape[1]
    M = K.shape[0]
    L = cholesky(K+sigma*1000*eye(M,M), lower=True)
#    L = cholesky(K, lower=True)
    return cho_solve((L, True), eye(M,M))