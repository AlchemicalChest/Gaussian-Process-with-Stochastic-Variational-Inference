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
from mpl_toolkits.mplot3d import Axes3D
from GPSVI.core.GPClassifier import GPClassifier

np.random.seed(0)

xdata, ydata = datasets.make_blobs(n_samples=1000, n_features=3, centers=4, \
                                   cluster_std=4.0, center_box=(-40.0,40.0))


from sklearn.cross_validation import train_test_split
xTr, xTe, yTr, yTe = train_test_split(xdata, ydata, test_size=0.50)

fig_name = 'Toy Blobs'
colors = ['r','g','b','c','m','y','k']
fig = plt.figure(fig_name)
ax = fig.add_subplot(221, projection='3d')
#ax = plt.subplot(221)
ax.set_title('Original Testing Data')
ax.scatter(xTe[:,0], xTe[:,1], xTe[:,2], c=[colors[y] for y in yTe], s=40)

#%%
clf_gpsvi = GPClassifier(xTr, yTr, None, None, \
                   alpha=0.5, max_iter=1000, num_inducing_points=100, \
                   kernel_type='rbf', kernel_args={'gamma':1.0/5}, \
                   learning_rate=0.01, verbose=2)
clf_gpsvi.fit()
pd = clf_gpsvi.predict(xTe)
print('SVI error = {}'.format(np.sum(len(np.where(pd != yTe)[0])) / float(xTe.shape[0])))
ax = fig.add_subplot(222, projection='3d')
#ax = plt.subplot(222)
ax.set_title('GP with Stochastic Variational Inference')
ax.scatter(xTe[:,0], xTe[:,1], xTe[:,2], c=[colors[y] for y in pd], s=40)

#%%
clf_svc = SVC()
clf_svc.fit(xTr, yTr)
pd = clf_svc.predict(xTe)
print('SVM error = {}'.format(np.sum(len(np.where(pd != yTe)[0])) / float(xTe.shape[0])))
#plt.figure(fig_name)
#ax = plt.subplot(223)
ax = fig.add_subplot(223, projection='3d')
ax.set_title('SVM with RBF Kernel')
ax.scatter(xTe[:,0], xTe[:,1], xTe[:,2], c=[colors[y] for y in pd], s=40)

#%%
clf_lr = LogisticRegression()
clf_lr.fit(xTr, yTr)
pd = clf_lr.predict(xTe)
print('Logistic error = {}'.format(np.sum(len(np.where(pd != yTe)[0])) / float(xTe.shape[0])))
#plt.figure(fig_name)
#ax = plt.subplot(224)
ax = fig.add_subplot(224, projection='3d')
ax.set_title('Logistic Regression')
ax.scatter(xTe[:,0], xTe[:,1], xTe[:,2], c=[colors[y] for y in pd], s=40)

