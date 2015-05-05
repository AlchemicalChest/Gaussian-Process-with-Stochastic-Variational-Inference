# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:30:05 2015

@author: Ziang
"""

from GPSVI.util.load_data import load_ogpc_tr
from GPSVI.util.load_data import load_ogpc_te
from GPSVI.core.GPClassifier import GPClassifier

#from sklearn.cross_validation import train_test_split
#from sklearn.linear_model import LogisticRegression

xTr, yTr = load_ogpc_tr()
xTe = load_ogpc_te()
#xTr, xTe, yTr, yTe = train_test_split(data, target, test_size=0.5)
#del data, target

clf_gp = GPClassifier(xTr, yTr, \
                   alpha=0.01, max_iter=100, num_inducing_points=1500, \
                   kernel_type='rbf', kernel_args={'gamma':0.001}, \
                   learning_rate=0.01, verbose=2)
clf_gp.fit()
#gp_score = clf_gp.score(xTe, yTe)
#print('Gaaussian Process:', gp_score)

pd = clf_gp.predict(xTe)

#clf = LogisticRegression(C=2.0)
#clf.fit(xTr, yTr)
#lr_score = clf.score(xTe, yTe)
#print('Logistic Regression: ', lr_score)