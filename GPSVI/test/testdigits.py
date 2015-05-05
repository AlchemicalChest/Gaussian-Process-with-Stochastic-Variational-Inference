# -*- coding: utf-8 -*-
"""
Created on Fri May  1 20:05:53 2015

@author: Ziang
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from GPSVI.core.GPClassifier import GPClassifier

np.random.seed(0)
data = datasets.load_digits()
xTr, xTe, yTr, yTe = train_test_split(data.data, data.target, test_size=0.50)

fig = plt.figure('Test Digits')

svd = TruncatedSVD(algorithm='randomized', n_components=3, tol=0.0)
svd.fit(xTr)
x = svd.transform(xTe)
ax = fig.add_subplot(121, projection='3d')
ax.scatter(x[:,0], x[:,1], x[:,2], c=yTe, cmap=matplotlib.cm.rainbow)



clf = GPClassifier(xTr, yTr, \
                   alpha=0.05, max_iter=300, num_inducing_points=200, \
                   kernel_type='rbf', kernel_args={'gamma':0.01}, \
                   learning_rate=0.01, verbose=2)
clf.fit()
pd = clf.predict(xTe)
gpsvi_score = clf.score(xTe, yTe)
print(gpsvi_score)

ax = fig.add_subplot(122, projection='3d')
ax.scatter(x[:,0], x[:,1], x[:,2], c=pd, cmap=matplotlib.cm.rainbow)
#clf_lr = LogisticRegression()
#clf_lr.fit(xTr, yTr)
#lr_score = clf_lr.score(xTe, yTe)
#print(lr_score)