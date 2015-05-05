# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:39:08 2015

@author: Ziang
"""
import numpy as np
from numpy import exp
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import pairwise_kernels

def compute_kernel(X1, X2=None, diag=False, ktype='rbf', n_jobs=1, **kwargs):
    """
    Compute kernel matrix
    
    Parameters
    ----------
    X1 : array [n_samples_a, n_features]
    
    X2 : array [n_samples_b, n_features]
    
    ktype : 'rbf', 'polynomial' or 'linear'
    
    Returns
    -------
    K : array [n_samples_a, n_samples_a] or [n_samples_a, n_samples_b]
        A kernel matrix K such that K_{i, j} is the kernel between the
        ith and jth vectors of the given matrix X, if Y is None.
        If Y is not None, then K_{i, j} is the kernel between the ith array
        from X and the jth array from Y.
        
    Notes
    -----
    parallel computing is not supported yet.
    """
    if ktype == 'rbf':
        gamma = 1.0 / X1.shape[1] if kwargs.get('gamma') is None else kwargs.get('gamma')
        if X2 is None and diag is True:
            return np.ones(X1.shape[0])
        return pairwise_kernels(X1, X2, metric=ktype, n_jobs=n_jobs, \
                                gamma=gamma)
    elif ktype == 'polynomial':
        gamma = 1.0 / X1.shape[1] if kwargs.get('gamma') is None else kwargs.get('gamma')
        degree = 3 if kwargs.get('degree') is None else kwargs.get('degree')
        coef0 = 1 if kwargs.get('coef0') is None else kwargs.get('coef0')
        if X2 is None and diag is True:
            return np.ones(X1.shape[0])+coef0**degree
        return pairwise_kernels(X1, X2, metric=ktype, n_jobs=n_jobs, \
                                gamma=gamma, degree=degree, coef0=coef0)
    elif ktype == 'linear':
        if X2 is None and diag is True:
            return np.array([np.sum(X1[i]**2) for i in range(X1.shape[0])], dtype=float)
        return pairwise_kernels(X1, X2, metric=ktype, n_jobs=n_jobs)
    else:
        raise Exception('Kernel type is not supported')
