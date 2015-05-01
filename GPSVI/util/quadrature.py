# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 23:06:27 2015

@author: Ziang
"""

from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.hermite import hermval
from math import pi
from math import sqrt

class GaussHermiteQuadrature:
    
    def __init__(self, degree):
        self.degree = degree
        self.hermite_samples, self.hermite_weights = hermgauss(degree)
        
    def get_samples(self):
        return self.hermite_samples
    
    def get_weights(self):
        return self.hermite_weights
        
    def get_hermite_polynomial(self, n, x):
        return hermval(x, [0]*n+[1])
    
    def quad(self, mu, sigma, func):
        hquad = 0
        N = len(self.hermite_samples)
        for i in range(N):
            x = self.hermite_samples[i]
            w = self.hermite_weights[i]
            hquad +=  w * func(sqrt(2)*sigma*x+mu)
        hquad /= sqrt(pi)
        return hquad
    
    def quad_v2(self, mu, sigma, func, *args):
        hquad = 0
        N = len(self.hermite_samples)
        for i in range(N):
            x = self.hermite_samples[i]
            w = self.hermite_weights[i]
            hquad +=  w * func(sqrt(2)*sigma*x+mu, *args)
        hquad /= sqrt(pi)
        return hquad