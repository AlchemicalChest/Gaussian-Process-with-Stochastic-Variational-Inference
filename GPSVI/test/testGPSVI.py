# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:24:26 2015

@author: Ziang
"""
import numpy as np
import time
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from GPSVI.core.GPClassifier import GPClassifier

np.random.seed(0)

data = datasets.load_digits()
xTr, xTe, yTr, yTe = train_test_split(data.data, data.target, test_size=0.50)
#del data

#%% compute benchmark
#t0 = time.time()
#clf = LogisticRegression(C=2.0, verbose=0)
#clf.fit(xTr, yTr)
#lr_score = clf.score(xTe, yTe)
#lr_t = time.time()-t0
#
#t0 = time.time()
#clf = SVC()
#clf.fit(xTr, yTr)
#svc_score = clf.score(xTe, yTe)
#svc_t = time.time()-t0

#%% run GP
#t0 = time.time()
clf = GPClassifier(xTr, yTr, \
                   alpha=0.5, max_iter=1, num_inducing_points=800, \
                   kernel_type='rbf', kernel_args={}, \
                   learning_rate=0.01, verbose=0)
clf.fit()
gpsvi_score = clf.score(xTe, yTe)
#gpsvi_t = time.time() -t0
#%% result

#print('LR      accuracy = {:.4f}, runtime = {:.4f}'.format(lr_score, lr_t))
#print('SVM     accuracy = {:.4f}, runtime = {:.4f}'.format(svc_score, svc_t))
#print('GPSVI   accuracy = {:.4f}, runtime = {:.4f}'.format(gpsvi_score, gpsvi_t))