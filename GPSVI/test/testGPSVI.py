# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:24:26 2015

@author: Ziang
"""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from GPSVI.core.GPClassifier import GPClassifier

np.random.seed(0)

data = datasets.load_digits()
xTr, xTe, yTr, yTe = train_test_split(data.data, data.target, test_size=0.50)

clf = GPClassifier(csr_matrix(xTr), yTr, \
                   alpha=0.5, max_iter=1, num_inducing_points=800, \
                   kernel_type='rbf', kernel_args={}, \
                   learning_rate=0.01, verbose=0)
clf.fit()
gpsvi_score = clf.score(xTe, yTe)