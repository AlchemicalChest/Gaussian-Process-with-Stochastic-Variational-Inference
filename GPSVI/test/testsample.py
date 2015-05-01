# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 22:26:23 2015

@author: Ziang
"""
import numpy as np
import scipy as sp
from sklearn import datasets
data = datasets.load_digits()
#np.random.seed(0)

from sklearn.cross_validation import train_test_split
xTr, xTe, yTr, yTe = train_test_split(data.data, data.target, test_size=0.50)

labels = np.unique(yTr)
C = len(labels)
labels_dist = []
total_size = xTr.shape[0]
for i in range(C):
    indices, = np.where(yTr==labels[i])
    labels_dist.append(indices)

def sample_x(N):
    if N > total_size:
        N = total_size
    num_samples = []
    for i in range(C):
        label_size = len(labels_dist[i])
        num = min(int(N*(label_size)/total_size*1.0), label_size)
        num_samples.append(num)
    while np.sum(num_samples) < N:
        i = (i + 1) % C
        label_size = len(labels_dist[i])
        if num_samples[i] < label_size:
            num_samples[i] += 1
    indices = []
    for i in range(C):
        label_size = len(labels_dist[i])
        indices += labels_dist[i][np.random.choice(range(label_size), size=num_samples[i], replace=False)].tolist()
    return indices

def kernel(A, B):
    l = 1
    import scipy
    from scipy.spatial.distance import cdist
    K = scipy.exp(- cdist(A, B) / (2 * l**2))
    K += np.diag([1]*K.shape[0])
    return K

h = xTr[sample_x(50),:]

k = kernel(h, h)
print(np.linalg.det(k))