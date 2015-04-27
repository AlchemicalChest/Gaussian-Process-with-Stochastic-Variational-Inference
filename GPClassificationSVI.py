# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 20:20:48 2015

@author: Ziang
"""
import math
from math import sqrt
import numpy as np
from numpy.random import randn
from numpy.random import normal
from numpy.random import choice
from scipy.stats import multivariate_normal
from scipy import linalg
from sympy import mpmath
from scipy.spatial.distance import cdist

from StochasticVariationalInference import StochasticVariationalInference
from util.approximation.GaussHermite import GaussHermite

class GPClassificationSVI(StochasticVariationalInference):
    def __init__(self, num_hidden_variables=10, quad_deg=10, hyper_param = 1, learning_rate=0.01, \
                       alpha=0.5, max_iter=10000, tolerance=1.0e-02, debug=False, mute=True, plot=False):
        StochasticVariationalInference.__init__(self, alpha, max_iter, tolerance, debug, mute, plot)
        self.num_hidden_variables = num_hidden_variables
        self.gaussHermite = GaussHermite(quad_deg)
        self.hyper_param = hyper_param
        self.learning_rate = learning_rate
        self.r0 = learning_rate
        self.mask = np.tril(np.ones((num_hidden_variables, num_hidden_variables))).flatten()      
    
    def score(self, x_tr, y_tr, x_te, y_te):
        self.labels = np.unique(y_tr)
        self.num_classes = len(self.labels)
        N,_ = x_tr.shape
        M = self.num_hidden_variables
        C = self.num_classes
        self.hidden_variables = np.array(x_tr[choice(range(N), size=M, replace=True), :])
        self.Kmm_inv = linalg.pinv(self.kernel(self.hidden_variables, self.hidden_variables))
        self.parameters = randn(M*C + M*M*C)
#        self.grads = [np.zeros(M*C + M*M*C), np.zeros(M*C + M*M*C)]
        self.infer_parameters(x_tr, y_tr, x_te, y_te)
    
    def fit(self, x_tr, y_tr):
        self.labels = np.unique(y_tr)
        self.num_classes = len(self.labels)
        N,_ = x_tr.shape
        M = self.num_hidden_variables
        C = self.num_classes
        self.hidden_variables = np.array(x_tr[choice(range(N), size=M, replace=True), :])
        self.Kmm_inv = linalg.pinv(self.kernel(self.hidden_variables, self.hidden_variables))
        self.parameters = randn(M*C + M*M*C)
#        self.grads = [np.zeros(M*C + M*M*C), np.zeros(M*C + M*M*C)]
        self.infer_parameters(x_tr, y_tr)
        
    def objective(self, z, x_tr, y_tr):
        val = 0
        (N,D) = x_tr.shape
        M = self.num_hidden_variables
        C = self.num_classes
        S = self.gaussHermite.degree
        h = self.hidden_variables
        Knn = self.kernel(x_tr, x_tr)
        Knm = self.kernel(x_tr, h)
        Kmm = self.kernel(h, h)
        A = Knm.dot(self.Kmm_inv)
        mu = []
        sigma = []
        for j in range(C):
            m = self.parameters[M*j:M*(j+1)].reshape(M,1)
            L = np.tril(self.parameters[M*C+M*M*j:M*C+M*M*(j+1)].reshape(M,M))
            mu.append(A.dot(m))
            sigma.append((Knn + A.dot(L.dot(L.T) - Kmm).dot(A.T)).diagonal())
        for i in range(N):
            y = y_tr[i]
            sample_sum = 0
            samples = self.gaussHermite.get_hermite_samples()
            weights = self.gaussHermite.get_hermite_weights()
            for k in range(S):    
                f = [samples[k]*abs(sigma[c][i]) + mu[c][i][0] for c in range(C)]
                exp_sum = np.sum([mpmath.exp(f[c]) for c in range(C)])
                sample_sum += weights[k]**C * mpmath.exp(f[y])/exp_sum 
            sample_sum /= sqrt(math.pi)
        val += mpmath.log(sample_sum)
        return float(val)
        
    def gradient(self, z, x_tr, y_tr):
        (N,D) = x_tr.shape
        M = self.num_hidden_variables
        C = self.num_classes
        S = self.gaussHermite.degree
        grad = np.zeros((M*C+M*M*C))
        
        h = self.hidden_variables # reference !!!
        u = [np.tril(np.array(self.parameters[M*C+M*M*j:M*C+M*M*(j+1)]).reshape(M,M)).dot(z) \
             + np.array(self.parameters[M*j:M*(j+1)]).reshape(M,1) for j in range(C)] # global variables
        Knn = self.kernel(x_tr, x_tr)
        Knm = self.kernel(x_tr, h)
#        Kmm_inv = linalg.pinv(self.kernel(h, h))
        Kmm_inv = self.Kmm_inv
        sigma = np.diag(Knn - Knm.dot(Kmm_inv).dot(Knm.T)) # cov for f
        mu = [Knm.dot(Kmm_inv).dot(u[j]) for j in range(C)] # mean for f
        A = Knm.dot(Kmm_inv)
        delta = [np.diag(1/self.parameters[M*C+M*M*c:M*C+M*M*(c+1)].reshape(M,M).diagonal()) for c in range(C)]

        for i in range(N):
            y = y_tr[i]
            sample_sum = 0
            samples = self.gaussHermite.get_hermite_samples()
            weights = self.gaussHermite.get_hermite_weights()
            for k in range(S):    
                f = [samples[k]*abs(sigma[i]) + mu[j][i][0] for j in range(C)]
                exp_sum = np.sum([mpmath.exp(f[j]) for j in range(C)])
                sample_sum += weights[k]**C * mpmath.exp(f[y])/exp_sum 
            sample_sum /= sqrt(math.pi)
            grad[M*y:M*(y+1)] += (1-sample_sum)*A[i, :]
            grad[M*C+M*M*y:M*C+M*M*(y+1)] += (1-sample_sum)*A[i, :].reshape(M,1).dot(z.T).flatten()

        for j in range(C):
            grad[M*j:M*(j+1)] -= Kmm_inv.dot(u[j]).flatten()
            grad[M*C+M*M*j:M*C+M*M*(j+1)] -= Kmm_inv.dot(u[j]).dot(z.T).flatten()
            grad[M*C+M*M*j:M*C+M*M*(j+1)] += delta[j].flatten()
        return grad
    
    def sample_z(self):
        return normal(0, 1, (self.num_hidden_variables, 1))
    
    def update(self, z, x_tr, y_tr, val, gradient, t):
        N,_ = x_tr.shape
        M = self.num_hidden_variables
        C = self.num_classes
        rho1 = self.learning_rate
        rho2 = self.learning_rate*0.1
        self.parameters[0:M*C] += rho1 * gradient[0:M*C]
        self.parameters[M*C: ] += rho2 * gradient[M*C: ]
        self.learning_rate = self.r0*(1+self.r0*0.05*t)**(-3/4)
#        print(self.learning_rate)
#        if np.random.random() > 0.9:
#            h = self.hidden_variables
#            self.hidden_variables[choice(range(M),size=M*0.1,replace=False)] = x_tr[choice(range(N),size=M*0.1,replace=False)]
#            self.Kmm_inv = linalg.pinv(self.kernel(self.hidden_variables, self.hidden_variables))
#            z0 = self.sample_z()
#            val_new = 0
#            for i in range(10):
#                val_new += self.objective(z0, x_tr, y_tr)
#            val_new /= 10
#            if val_new < val*0.7:
#                self.hidden_variables = h
        
    def predict_prob(self, x_te):
        (N,D) = x_te.shape
        M = self.num_hidden_variables
        C = self.num_classes
        S = self.gaussHermite.degree
        h = self.hidden_variables
        probs = np.zeros((N, C))
        Knn = self.kernel(x_te, x_te)
        Knm = self.kernel(x_te, h)
        Kmm = self.kernel(h, h)
        A = Knm.dot(linalg.pinv(Kmm))
        mu = []
        sigma = []
        for j in range(C):
            m = self.parameters[M*j:M*(j+1)].reshape(M,1)
            L = np.tril(self.parameters[M*C+M*M*j:M*C+M*M*(j+1)].reshape(M,M))
            mu.append(A.dot(m))
            sigma.append((Knn + A.dot(L.dot(L.T) - Kmm).dot(A.T)).diagonal())
        for i in range(N):
            sample_sum = 0
            samples = self.gaussHermite.get_hermite_samples()
            weights = self.gaussHermite.get_hermite_weights()
            for j in range(C):
                for k in range(S):    
                    f = [samples[k]*abs(sigma[c][i]) + mu[c][i][0] for c in range(C)]
                    exp_sum = np.sum([mpmath.exp(f[c]) for c in range(C)])
                    sample_sum += weights[k]**C * mpmath.exp(f[j])/exp_sum 
                sample_sum /= sqrt(math.pi)
                probs[i,j] = sample_sum
#        exponential = lambda x : mpmath.exp(x)
#        for i in range(N):
#            f = [self.gaussHermite.hermite_quad(mu[j][i][0], sqrt(sigma[j][i]**2), func=exponential) for j in range(C)]
#            exp_sum = np.sum(f)
#            probs[i, :] = [float(f[j]/exp_sum) for j in range(C)]
        return probs
    
    def predict(self, X):
        (n, _) = X.shape
        probs = self.predict_prob(X)
        return np.array([self.labels[np.argmax(probs[i, :])] for i in range(n)])
    
    def kernel(self, A, B):
        l = self.hyper_param
        return np.exp(- cdist(A, B) / (2 * l**2))
        