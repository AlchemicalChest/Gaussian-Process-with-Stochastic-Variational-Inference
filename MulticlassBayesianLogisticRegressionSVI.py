# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:08:08 2015

@author: Ziang
"""
import numpy as np
from scipy import linalg
from scipy import sparse
from scipy.stats import norm
from sympy import mpmath
from StochasticVariationalInference import StochasticVariationalInference

class MulticlassBayesianLogisticRegressionSVI(StochasticVariationalInference):
    
    def fit(self, X, Y):
        self.labels = np.sort(np.unique(Y))
        _, self.num_dimension = X.shape
        self.num_class = len(self.labels)
        self.var_mean = np.zeros((self.num_dimension * self.num_class,1))
        if self.num_dimension*self.num_class < self.dlim:
            self.var_cov_sqrt = np.triu(np.random.rand(self.num_dimension*self.num_class, self.num_dimension*self.num_class)) # cov = S*S.T
        else:
            self.var_cov_sqrt = sparse.diags(np.random.rand(self.num_dimension*self.num_class), 0).tocsr()
        self.block_mask = linalg.block_diag(*[np.ones((self.num_dimension, self.num_dimension))]*self.num_class)
        self.rho = 0.01
        self.infer_parameters(X, Y)
        
    
    def objective(self, z, X, Y):
        w = np.add(self.var_cov_sqrt.dot(z), self.var_mean) # Cd x 1
        gradient = np.zeros((w.shape), dtype=float) # Cd x 1
        for n in range(X.shape[0]):
            if sparse.issparse(X):
                x = X[n, :].toarray()
            else:
                x = np.array(X[n, :]).reshape(1, self.num_dimension)
            y = Y[n]
            g = np.zeros((w.shape), dtype=float)
            g[self.num_dimension*y:self.num_dimension*(y+1),:] = x.T
            a = []            
            b = 0
            for c in range(self.num_class):
                a.append(x * mpmath.exp(x.dot(w[self.num_dimension*c:self.num_dimension*(c+1),:])[0,0]))
                b += mpmath.exp(x.dot(w[self.num_dimension*c:self.num_dimension*(c+1),:])[0,0])
            for c in range(self.num_class):
                g[self.num_dimension*c:self.num_dimension*(c+1),:] -= [[float(h)] for h in (a[c]/b).T]
            gradient += g
        var_cov_sqrt_diag = self.var_cov_sqrt.diagonal().reshape((self.num_dimension*self.num_class,1))
        c = w / (np.multiply(var_cov_sqrt_diag, var_cov_sqrt_diag) + np.multiply(self.var_mean, self.var_mean))
        gradient -= c
        return 0, gradient
    
    def sample_z(self):
        return np.random.normal(0, 1, (self.num_dimension*self.num_class, 1))
    
    def update(self, z, val, gradient, t):
        if t % 5000 == 0:
            self.rho *= 0.95
#        self.rho = self.rho * (1 + self.rho * 0.0000001 * t) ** (-0.75)
        rho_1 = self.rho
        rho_2 = self.rho*0.1           
        
        m_new = np.add(self.var_mean, rho_1 * gradient)
        if sparse.issparse(self.var_cov_sqrt):
            var_cov_sqrt_diag = self.var_cov_sqrt.diagonal().reshape((self.num_dimension*self.num_class,1))
            var_cov_sqrt_diag = np.add(var_cov_sqrt_diag, rho_2 * (np.add(np.multiply(gradient, z), 1.0/var_cov_sqrt_diag)))
            S_new = sparse.diags(self.var_cov_sqrt_diag.reshape(self.num_dimension), 0)
        else:
            S_new = np.add(self.var_cov_sqrt, rho_2 * (np.triu(gradient.dot(z.T)) + np.diag(1.0/np.diag(self.var_cov_sqrt))))
            S_new = np.multiply(S_new, self.block_mask)
        self.var_mean = m_new
        self.var_cov_sqrt = S_new
        
    def predict_prob(self, X):
        (N,_) = X.shape
        probs = np.zeros((N, self.num_class))
        mean = self.get_mean()
        cov = self.get_cov()
        for n in range(N):
            if sparse.issparse(X):
                x = X[n,:]
            else:
                x = np.array(X[n,:])
            for c in range(self.num_class):
                a = x.dot(mean[self.num_dimension*c:self.num_dimension*(c+1), :])
                if sparse.issparse(cov):
                    b = np.sqrt(1 + x.dot(cov[self.num_dimension*c:self.num_dimension*(c+1), self.num_dimension*c:self.num_dimension*(c+1)]).dot(x.T).toarray())
                else:
                    b = np.sqrt(1 + x.dot(cov[self.num_dimension*c:self.num_dimension*(c+1), self.num_dimension*c:self.num_dimension*(c+1)]).dot(x.T))
                t = norm.cdf(float(a/b))
                probs[n, c] = t
        return probs
    
    def predict(self, X):
        (n, _) = X.shape
        probs = self.predict_prob(X)
        return np.array([self.labels[np.argmax(probs[i, :])] for i in range(n)])