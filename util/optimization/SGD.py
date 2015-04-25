# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:19:42 2015

@author: Ziang
"""
from numpy.random import choice
def stochasticGradientDecent(X, y, w0, alpha, objFunc, updateFunc, max_step=1000, tolerance=10e-4):
    w = w0
    N,_ = X.shape
    step = 0
    while step < max_step:
        sampleIndex = choice(range(N), size=max(1, alpha*N), replace=False)
        val, grad = objFunc(X[sampleIndex, :], y[sampleIndex], w)
        w = updateFunc(grad, w, step)
    return w0