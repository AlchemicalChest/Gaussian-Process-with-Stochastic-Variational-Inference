# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 22:44:05 2015

@author: Ziang
"""
from sympy import mpmath
from math import sqrt
from math import pi
from numpy import inf
from scipy.integrate import quad

from GaussHermite import GaussHermite
#from GaussHermiteQuadrature import GaussHermiteQuadrature

mu = 1
sigma = 2

sigmoid = lambda x : 1/(1+mpmath.exp(-x))
polynomial = lambda x : 2*x**6+4*x**9+x+2
exponential = lambda x : mpmath.exp(x)
normalpdf = lambda x : 1 / (sqrt(2*pi)*sigma) * mpmath.exp(-(x-mu)**2/(2*sigma**2))


func = lambda x: polynomial(x) * normalpdf(x)
print('Benchmark =', quad(func, -inf, inf))

gh = GaussHermite(20)
print('GaussHermite =', gh.hermite_quad(mu, sigma, polynomial))