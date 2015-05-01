# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:09:51 2015

@author: Ziang
"""
#from numpy.polynomial.hermite import hermval
#from numpy.polynomial.hermite import hermgauss

from numpy import inf
#from math import factorial
from math import sqrt
from math import pi
from sympy import mpmath
from scipy.integrate import quad
from GaussHermiteQuadrature import GaussHermiteQuadrature as gaussquad
#def H(n, x):
#    return hermval(x, [0]*n+[1])
#def hermite_weight(n, x):
#    return pow(2,n-1)*factorial(n)*sqrt(pi)/(pow(n,2)*pow(H(n-1, x), 2))
mu = 1
sigma = 2

sigmoid = lambda x : 1/(1+mpmath.exp(-x))
polynomial = lambda x : 2*x**6+4*x**9+x+2
exponential = lambda x : mpmath.exp(x)
normalpdf = lambda x : 1 / (sqrt(2*pi)*sigma) * mpmath.exp(-(x-mu)**2/(2*sigma**2))

#N = 10
#x,u = hermgauss(N)
#z = x
#
#
#hquad = 0
#for i in range(N):
#    w = hermite_weight(N, x[i])
#    hquad += w * polynomial(sqrt(2)*sigma*x[i]+mu)
#hquad /= sqrt(pi)
#
#print('hquad=', hquad)

hquad = gaussquad.hermite_quad(mu, sigma, func=exponential, N=20)
print('hquad=', hquad)

#test = gaussquad.hermite_quad_test(mu, sigma, func=polynomial, N=10)
#print('test=', test)

func = lambda x: exponential(x) * normalpdf(x)
print('quad =', quad(func, -inf, inf))
