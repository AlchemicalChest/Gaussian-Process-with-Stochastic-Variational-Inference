# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:24:55 2015

@author: Ziang
"""
import time
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from GPSVI.core.GPClassifier import GPClassifier

np.random.seed(0)

data = datasets.fetch_20newsgroups_vectorized()

xTr, xTe, yTr, yTe = train_test_split(data.data, data.target, test_size=0.50)
del data

t0 = time.time()
clf_gp = GPClassifier(xTr, yTr, \
                   alpha=0.1, max_iter=500, num_inducing_points=800, \
                   kernel_type='rbf', kernel_args={'gamma':1.0}, \
                   learning_rate=0.01, verbose=2)
clf_gp.fit()
gp_score = clf_gp.score(xTe, yTe)
gp_t = time.time()-t0

t0 = time.time()
clf = LogisticRegression(C=2.0)
clf.fit(xTr, yTr)
lr_score = clf.score(xTe, yTe)
lr_t = time.time()-t0

t0 = time.time()
clf = SVC()
clf.fit(xTr, yTr)
svc_score = clf.score(xTe, yTe)
svc_t = time.time()-t0