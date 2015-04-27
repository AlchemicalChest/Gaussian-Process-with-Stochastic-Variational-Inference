# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:24:26 2015

@author: Ziang
"""
import numpy as np
import time

np.random.seed(0)

#%% load data
from sklearn import datasets
data = datasets.load_iris()
x_raw = data.data
y_raw = data.target
del data

from sklearn.cross_validation import train_test_split
x_tr, x_te, y_tr, y_te = train_test_split(x_raw, y_raw, test_size=0.50)
del x_raw, y_raw

#%% compute benchmark
from sklearn.linear_model import LogisticRegression
t0 = time.time()
clf = LogisticRegression(C=2.0, verbose=0)
clf.fit(x_tr, y_tr)
benchmark = clf.predict(x_te)
(n,_) = x_te.shape
benchmark_error = np.sum(len(np.where(benchmark != y_te)[0])) / float(n)
t_benchmark = time.time()-t0
del n, clf, t0

##%% run multiclass svi
#from MulticlassBayesianLogisticRegressionSVI import MulticlassBayesianLogisticRegressionSVI
#t0 = time.time()
#clf = MulticlassBayesianLogisticRegressionSVI(sample=0.2, max_iter=10000, debug=True, mute=False)
#clf.fit(X_tr, y_tr)
#msvi_pd = clf.predict(X_te)
#(n,_) = X_te.shape
#msvi_error = np.sum(len(np.where(msvi_pd != y_te)[0])) / float(n)
#t_msvi = time.time() - t0
#del n, clf, t0
#
##%% run one vs rest svi
#from BayesianLogisticRegressionSVI import BayesianLogisticRegressionSVI
#t0 = time.time()
#labels = np.sort(np.unique(y_tr))
#(n,_) = X_te.shape
#ssvi_prob = np.zeros((n, len(labels)))
#for label in labels:
#    y = np.array(y_tr)
#    y[y != label] = -11
#    y[y == label] = 1
#    y[y == -11] = 0
#    clf = BayesianLogisticRegressionSVI(sample=0.2, max_iter=1000, debug=True, mute=False)
#    clf.fit(X_tr, y)
#    ssvi_prob[:, label] = clf.predict_prob(X_te)
#ssvi_pd = np.argmax(ssvi_prob, 1)
#ssvi_error = np.sum(len(np.where(ssvi_pd != y_te)[0])) / float(n)
#t_ssvi = time.time() - t0

#%% run GP
from GPClassificationSVI import GPClassificationSVI
t0 = time.time()
clf = GPClassificationSVI(alpha=0.1, max_iter=2000, debug=True, mute=False, plot=True, \
                          learning_rate=0.01, num_hidden_variables=30, quad_deg=20, hyper_param=1)
#clf.fit(x_tr, y_tr, num_hidden_variables=10, num_samples=20, hyper_param = 1)
#clf.score(x_tr, y_tr, x_te, y_te)
clf.fit(x_tr, y_tr)
gpsvi_pd = clf.predict(x_te)
gpsvi_probs = clf.predict_prob(x_te)
(n,_) = x_te.shape
gpsvi_error = np.sum(len(np.where(gpsvi_pd != y_te)[0])) / float(n)
t_gpsvi = time.time() -t0
del n, clf, t0
#%% result

print('Benchmark      accuracy = {:.4f}, runtime = {:.4f}'.format(1 - benchmark_error, t_benchmark))
#print('Multiclass SVI accuracy = {:.4f}, runtime = {:.4f}'.format(1 - msvi_error, t_msvi))
#print('OneVsRest SVI  accuracy = {:.4f}, runtime = {:.4f}'.format(1 - ssvi_error, t_ssvi))
print('GPSVI          accuracy = {:.4f}, runtime = {:.4f}'.format(1 - gpsvi_error, t_gpsvi))