# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 22:37:55 2015

@author: Ziang
"""
import numpy as np
from sympy import mpmath
from scipy import sparse
from scipy.stats import norm
from StochasticVariationalInference import StochasticVariationalInference

class BayesianLogisticRegressionSVI(StochasticVariationalInference):
    
    def fit(self, X, Y):
        _, self.D = X.shape
        self.m = np.zeros((self.D,1))
        if self.D < self.dlim:
            self.S = np.triu(np.random.rand(self.D, self.D)) # cov = S*S.T
        else:
            self.S = sparse.diags(np.random.rand(self.D), 0).tocsr()
        self.rho = 0.1
        self.infer_parameters(X, Y) 
        
    def sample_z(self):
        return np.random.normal(0, 1, (self.D, 1))
    
    def update(self, z, val, grad, t):
        if t % 100 == 0:
            self.rho *= 0.95
        (d,_) = self.S.shape
        self.m = np.add(self.m, self.rho * grad)
        if sparse.issparse(self.S):
            S_diag = np.array(self.S.diagonal()).reshape((d,1))
            S_diag = np.add(S_diag, 0.1 * self.rho * (np.add(np.multiply(grad, z), 1.0/S_diag)))
            self.S.setdiag(self.S_diag.reshape(d))
        else:
            self.S = np.triu(np.add(self.S, 0.1 * self.rho * (grad.dot(z.T) + np.diag(1.0/np.diag(self.S)))))

    def objective(self, z, X, y):
        (N,_) = X.shape
        w = np.add(self.S.dot(z), self.m) # O(d^2)
        sigmoid = lambda x: float(1/(1+mpmath.exp(-float(x))))
#        val = 0
        grad = np.zeros((self.D, 1), dtype=float)
        for i in range(N):
            if sparse.issparse(X):
                x = X[i, :]
            else:
                x = np.array([X[i, :]])
            t = x.dot(w)
            f = sigmoid(t)
            if sparse.issparse(X):
                grad = np.add(grad, (x * (y[i] - f)).toarray().T)
            else:
                grad = np.add(grad, np.array(x * (y[i] -f)).T)
        S_diag = self.S.diagonal().reshape((self.D,1))
        L = w / (np.multiply(S_diag, S_diag) + np.multiply(self.m, self.m))
        grad -= L
        return 0, grad
        
    def predict(self, X):
        probs = self.predict_prob(X)
        return np.array([0 if p < 0.5 else 1 for p in probs ])
    
    def predict_prob(self, X):
        (n,_) = X.shape
        probs = np.zeros(n)   
        m = self.get_mean()
        cov = self.get_cov()        
        for i in range(n):
            if sparse.issparse(X):
                x = X[i,:]
                b = np.sqrt(1+x.dot(cov).dot(x.T).toarray())
            else:
                x = np.array(X[i,:])
                b = np.sqrt(1+x.dot(cov).dot(x.T))
            a = x.dot(m)            
            t = norm.cdf(float(a/b))
            probs[i] = t
        return probs
        