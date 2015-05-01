# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:37:22 2015

@author: Ziang
"""
import numpy as np
from util.kernel import compute_kernel
from util.inverse import cho_inverse

from sklearn import datasets
data = datasets.load_digits()

from sklearn.cross_validation import train_test_split
xTr, xTe, yTr, yTe = train_test_split(data.data, data.target, test_size=0.1)

M = 1600
h = xTr[np.random.choice(range(xTr.shape[0]), size=M, replace=False),:]

K = compute_kernel(h,h)
K_inv = cho_inverse(K)

print(K.dot(K_inv))