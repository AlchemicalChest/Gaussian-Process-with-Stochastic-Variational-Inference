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
from GPClassificationSVI import GPClassificationSVI

np.random.seed(0)

xdata, ydata = datasets.make_blobs(n_samples=400, centers=3)


from sklearn.cross_validation import train_test_split
x_tr, x_te, y_tr, y_te = train_test_split(xdata, ydata, test_size=0.50)


colors = ['r','g','b']
plt.figure(2)
ax = plt.subplot(221)
ax.set_title('Original Testing Data')
ax.scatter(x_te[:,0], x_te[:,1], c=[colors[y] for y in y_te], s=40)

clf_gpsvi = GPClassificationSVI(num_hidden_variables=20, quad_deg=20, hyper_param = 1, \
                          learning_rate=0.01, alpha=0.1, max_iter=500, \
                          debug=True, mute=False, plot=True)
#clf.score(x_tr, y_tr, x_te, y_te)
clf_gpsvi.fit(x_tr, y_tr)
pd = clf_gpsvi.predict(x_te)
print('SVI error = {}'.format(np.sum(len(np.where(pd != y_te)[0])) / float(x_te.shape[0])))
plt.figure(2)
ax = plt.subplot(222)
ax.set_title('GP with Stochastic Variational Inference')
ax.scatter(x_te[:,0], x_te[:,1], c=[colors[y] for y in pd], s=40)

clf_svc = SVC()
clf_svc.fit(x_tr, y_tr)
pd = clf_svc.predict(x_te)
print('SVM error = {}'.format(np.sum(len(np.where(pd != y_te)[0])) / float(x_te.shape[0])))
plt.figure(2)
ax = plt.subplot(223)
ax.set_title('SVM with RBF Kernel')
ax.scatter(x_te[:,0], x_te[:,1], c=[colors[y] for y in pd], s=40)

clf_lr = LogisticRegression()
clf_lr.fit(x_tr, y_tr)
pd = clf_lr.predict(x_te)
print('Logistic error = {}'.format(np.sum(len(np.where(pd != y_te)[0])) / float(x_te.shape[0])))
plt.figure(2)
ax = plt.subplot(224)
ax.set_title('Logistic Regression')
ax.scatter(x_te[:,0], x_te[:,1], c=[colors[y] for y in pd], s=40)

