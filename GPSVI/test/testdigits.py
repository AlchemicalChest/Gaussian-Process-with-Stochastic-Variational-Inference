# -*- coding: utf-8 -*-
"""
Created on Fri May  1 20:05:53 2015

@author: Ziang
"""

import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from GPSVI.core.GPClassifier import GPClassifier

np.random.seed(0)

data = datasets.load_digits()
xTr, xTe, yTr, yTe = train_test_split(data.data, data.target, test_size=0.50)
print(data.data.shape)

clf = GPClassifier(xTr, yTr, \
                   alpha=0.5, max_iter=300, num_inducing_points=400, \
                   kernel_type='polynomial', kernel_args={'gamma':1.0/800}, \
                   learning_rate=0.01, verbose=3)
clf.fit()
gpsvi_score = clf.score(xTe, yTe)
print(gpsvi_score)