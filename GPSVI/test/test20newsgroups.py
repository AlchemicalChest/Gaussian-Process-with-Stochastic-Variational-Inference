# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:24:55 2015

@author: Ziang
"""
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from GPSVI.core.GPClassifier import GPClassifier

np.random.seed(0)

data = datasets.fetch_20newsgroups_vectorized()

xTr, xTe, yTr, yTe = train_test_split(data.data, data.target, test_size=0.80)

svd = TruncatedSVD(algorithm='randomized', n_components=3, tol=0.0)
svd.fit(xTr)
x = svd.transform(xTr)
fig = plt.figure('Show data')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0], x[:,1], x[:,2], c=yTr, cmap=matplotlib.cm.rainbow)


t0 = time.time()
clf_lr = LogisticRegression(C=2.0)
clf_lr.fit(xTr, yTr)
lr_score = clf_lr.score(xTe, yTe)
lr_t = time.time()-t0

t0 = time.time()
clf_svc = SVC()
clf_svc.fit(xTr, yTr)
svc_score = clf_svc.score(xTe, yTe)
svc_t = time.time()-t0


t0 = time.time()
clf_gp = GPClassifier(xTr, yTr, \
                   alpha=0.05, max_iter=100, num_inducing_points=1500, \
                   kernel_type='rbf', kernel_args={'gamma':2.0}, \
                   learning_rate=0.01, verbose=2)
clf_gp.fit()
gp_score = clf_gp.score(xTe, yTe)
gp_t = time.time()-t0