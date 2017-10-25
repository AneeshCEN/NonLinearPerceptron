#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 12:02:49 2017

@author: aneesh
"""
import numpy as np

from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
np.random.seed(0)

X, y = make_circles(n_samples=800, factor=.3, noise=.05)

plt.scatter(X[y==1,0], X[y==1,1], c='r', label='class 1')
plt.scatter(X[y==0,0], X[y==0,1], c='g', label='class 2')

plt.xlabel('X - axis')
plt.ylabel('Y - axis')
plt.title('Not linearly separable')

plt.legend(loc='upper right')
plt.show()