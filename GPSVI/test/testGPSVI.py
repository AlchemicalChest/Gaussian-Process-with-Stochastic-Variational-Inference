"""
Created on Sat Apr 11 20:24:26 2015

@author: Ziang
"""
import numpy as np
from sklearn import datasets
from GPSVI.core.GPClassifier import GPClassifier

np.random.seed(0)

data_tr = datasets.fetch_20newsgroups_vectorized(subset='train')
data_te = datasets.fetch_20newsgroups_vectorized(subset='test')

print(data_tr.data.shape)
print(data_te.data.shape)

clf = GPClassifier(data_tr.data, data_tr.target, \
                   alpha=0.2, max_iter=1, num_inducing_points=1000, \
                   kernel_type='rbf', kernel_args={'gamma':1.0}, \
                   learning_rate=0.01, verbose=0)
clf.fit()
gpsvi_score = clf.score(data_te.data, data_te.target)
print(gpsvi_score)