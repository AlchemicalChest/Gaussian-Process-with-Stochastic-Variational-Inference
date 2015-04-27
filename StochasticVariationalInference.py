# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 18:16:36 2015

@author: Ziang
"""
import numpy as np
from time import time
class StochasticVariationalInference:

    def __init__(self, alpha=0.5, max_iter=10000, tolerance=1.0e-02, debug=False, mute=True, plot=False):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.debug = debug
        self.mute = mute
        self.plot = plot
        
    def sample_z(self):
        raise NotImplementedError
    
    def update(self, z, x, y, val, gradient, t):
        raise NotImplementedError
        
    def objective(self, z, X, Y):
        raise NotImplementedError
        
    def gradient(self, z, x, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
    
    def predict_prob(self, X):
        raise NotImplementedError
    
    def fit(self, X, Y):
        raise NotImplementedError
    
    def infer_parameters(self, x_tr, y_tr, x_te=None, y_te=None):
        # initialize parameters
        (N,D) = x_tr.shape # dimensions
        t = 0
        z = self.sample_z()
        index = np.random.choice(range(N), size=max(1, self.alpha*N), replace=True)
        grad = self.gradient(z, x_tr[index, :], y_tr[index])
        val = self.objective(z, x_tr[index, :], y_tr[index])
        # for debugging        
        if self.debug:
            if self.plot:
                from matplotlib import pyplot as plt
                plt.figure(1)
                plt.ion()
                xdata = [t] # steps
                vals = [val]
                ydata1 = [np.sum(len(np.where(self.predict(x_tr) != y_tr)[0])) / float(x_tr.shape[0])]
                if x_te is not None and y_te is not None:
                    ydata2 = [np.sum(len(np.where(self.predict(x_te) != y_te)[0])) / float(x_tr.shape[0])]
                    plt.subplot(211)
                    lines1 = plt.plot(xdata, ydata1,'g', xdata, ydata2, 'b')
                    
                else:
                    plt.subplot(211)
                    lines1 = plt.plot(xdata, ydata1,'g')
                plt.ylim(0, 1)
                plt.grid(True)
                plt.subplot(212)
                lines2 = plt.plot(xdata, vals, 'r')
                plt.ylim(np.min(vals)-1, np.max(vals)+1)
                plt.grid(True)
            
        # perform stochastic gradient descent (max(func))
        t0 = time()
        while t < self.max_iter and np.linalg.norm(grad) > self.tolerance:
            t = t + 1
            self.update(z, x_tr[index, :], y_tr[index], val, grad, t)
            # sample z
            z = self.sample_z()
            # sample X
            index = np.random.choice(range(N), size=max(1, self.alpha*N), replace=True)
            grad = self.gradient(z, x_tr[index, :], y_tr[index])            
            val = self.objective(z, x_tr[index, :], y_tr[index])
            # for debugging
            if self.debug:
                if self.plot:
                    xdata.append(t)
                    ydata1.append(np.sum(len(np.where(self.predict(x_tr) != y_tr)[0])) / float(x_tr.shape[0]))
                    lines1[0].set_data(xdata, ydata1)
                    if x_te is not None and y_te is not None:
                        ydata2.append(np.sum(len(np.where(self.predict(x_te) != y_te)[0])) / float(x_te.shape[0]))
                        lines1[1].set_data(xdata, ydata2)
                    plt.subplot(211)
                    plt.xlim(xmax=np.max(xdata))
                    plt.draw()
                    vals.append(val)
                    lines2[0].set_data(xdata, vals)
                    plt.subplot(212)
                    plt.xlim(xmax=np.max(xdata))
                    plt.ylim(np.min(vals)-1, np.max(vals)+1)
                    plt.draw()
                if not self.mute and self.plot:
                    if x_te!=None and y_te!=None:
                        print('Iter {:5}: \ntr_error = {} \nte_error = {}'.format(t, ydata1[-1], ydata2[-1]))
                    else:
                        print('Iter {:5}: tr_error = {}'.format(t, ydata1[-1]))
                if not self.mute and not self.plot:
                    print('Iter {:5}: val = {}'.format(t, val))
            if not self.mute:
                print('{:6d} iterations, val={:.16f}, time = {:.5f}s'.format(t, val, time()-t0))