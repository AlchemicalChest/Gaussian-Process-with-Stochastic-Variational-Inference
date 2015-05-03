# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 00:52:00 2015

@author: Ziang
"""
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from GPSVI.core.GPClassifier import GPClassifier
from sklearn.cross_validation import train_test_split

np.random.seed(0)

#xdata, ydata = datasets.make_moons(n_samples=800, noise=0.1)
xdata, ydata = datasets.make_circles(n_samples=800, noise=0.1, factor=0.5)

xTr, xTe, yTr, yTe = train_test_split(xdata, ydata, test_size=0.50)
x_min, x_max = xdata[:, 0].min() - .5, xdata[:, 0].max() + .5
y_min, y_max = xdata[:, 1].min() - .5, xdata[:, 1].max() + .5
h = 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#%% draw dataset
plt.figure('Decision Boundary')
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#d7191c', '#2b83ba'])
#plt.subplot(221)
#plt.scatter(xTe[:, 0], xTe[:, 1], c=yTe, cmap=cm_bright, alpha=0.5)
#plt.scatter(xTe[:, 0], xTe[:, 1], c=yTe, cmap=cm_bright)
#plt.xlim(xx.min(), xx.max())
#plt.ylim(yy.min(), yy.max())

#%% gp svi
clf_gp = GPClassifier(xTr, yTr, \
                      alpha=0.7, max_iter=500, num_inducing_points=200, \
                      kernel_type='rbf', kernel_args={'gamma':2.0}, \
                      learning_rate=0.01, verbose=3)
clf_gp.fit()
score = clf_gp.score(xTe, yTe)
zz = clf_gp.predict_proba(np.c_[xx.ravel(), yy.ravel()])
zz = (zz[:,1] / np.sum(zz, axis=1)).reshape(xx.shape)
plt.figure('Decision Boundary')
ax = plt.subplot(2,2,1)
ax.contourf(xx, yy, zz, cmap=cm, alpha=.8)
ax.scatter(xTr[:, 0], xTr[:, 1], c=yTr, cmap=cm_bright, alpha=0.5)
ax.scatter(xTe[:, 0], xTe[:, 1], c=yTe, cmap=cm_bright, alpha=1)
hx, hy = clf_gp.get_inducing_points()
ax.scatter(hx[:, 0], hx[:, 1], marker='+', c=hy, cmap=cm_bright)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_title('GP SVI')
ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')

#%% train benchmark classifier
names = ['RBF SVM', 'Logistic Regression', 'Random Forest']
classifiers = [SVC(), LogisticRegression(), RandomForestClassifier()]
i = 2
for name, clf in zip(names, classifiers):
    clf.fit(xTr, yTr)
    score = clf.score(xTe, yTe)
    if hasattr(clf, "decision_function"):
        zz = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        zz = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    zz = zz.reshape(xx.shape)
    plt.figure('Decision Boundary')
    ax = plt.subplot(2,2,i)
    ax.contourf(xx, yy, zz, cmap=cm, alpha=.8)
    ax.scatter(xTr[:, 0], xTr[:, 1], c=yTr, cmap=cm_bright, alpha=0.5)
    ax.scatter(xTe[:, 0], xTe[:, 1], c=yTe, cmap=cm_bright, alpha=1)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
    i += 1