# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 22:27:54 2015

@author: Ziang
"""
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from GPSVI.core.GPClassifier import GPClassifier

np.random.seed(0)

xdata, ydata = datasets.make_circles(n_samples=500, noise=0.1, factor=0.5)


from sklearn.cross_validation import train_test_split
x_tr, x_te, y_tr, y_te = train_test_split(xdata, ydata, test_size=0.50)


colors = ['r','g','b']
plt.figure('Toy Circle')
ax = plt.subplot(221)
ax.set_title('Original Testing Data')
ax.scatter(x_te[:,0], x_te[:,1], c=[colors[y] for y in y_te], s=40)

clf = GPClassifier(x_tr, y_tr, None, None, \
                   alpha=0.1, max_iter=500, num_inducing_points=180, \
                   kernel_type='rbf', kernel_args={'gamma':80.0}, \
                   learning_rate=0.01, verbose=2)
clf.fit()
pd = clf.predict(x_te)
print('SVI error = {}'.format(np.sum(len(np.where(pd != y_te)[0])) / float(x_te.shape[0])))
plt.figure('Toy Circle')
ax = plt.subplot(222)
ax.set_title('GP with Stochastic Variational Inference')
ax.scatter(x_te[:,0], x_te[:,1], c=[colors[y] for y in pd], s=40)

clf = SVC()
clf.fit(x_tr, y_tr)
pd = clf.predict(x_te)
print('SVM error = {}'.format(np.sum(len(np.where(pd != y_te)[0])) / float(x_te.shape[0])))
plt.figure('Toy Circle')
ax = plt.subplot(223)
ax.set_title('SVM with RBF Kernel')
ax.scatter(x_te[:,0], x_te[:,1], c=[colors[y] for y in pd], s=40)

clf = LogisticRegression()
clf.fit(x_tr, y_tr)
pd = clf.predict(x_te)
print('Logistic error = {}'.format(np.sum(len(np.where(pd != y_te)[0])) / float(x_te.shape[0])))
plt.figure('Toy Circle')
ax = plt.subplot(224)
ax.set_title('Logistic Regression')
ax.scatter(x_te[:,0], x_te[:,1], c=[colors[y] for y in pd], s=40)

