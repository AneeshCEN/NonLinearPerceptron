#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 12:02:49 2017

@author: aneesh
"""
import numpy as np

from sklearn.datasets import make_circles, make_blobs, make_moons
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
np.random.seed(0)

#X,y = make_circles(n_samples=800,factor=0.3, noise=0.05)

X,y = make_moons(n_samples=300, shuffle=True, noise=0.3, random_state=None)

#X, y = make_blobs(n_samples=1000,n_features=2,centers=2,cluster_std=1.0,random_state=20)

def feature_mapping(x1, x2, degree):
    phi_x = np.array([])
    for i in range(1, degree+1):
        for j in range(i+1):
            mapped_feature=(x1**(i-j))*(x2**(j))
            if len(phi_x) == 0:
                phi_x = mapped_feature
            else:
                phi_x = np.column_stack((phi_x,mapped_feature))
                
    return phi_x
    
    

plt.scatter(X[y==1,0], X[y==1,1], c='r', label='class 1')
plt.scatter(X[y==0,0], X[y==0,1], c='g', label='class 2')

phi_x = feature_mapping(X[:,0], X[:,1], 10)

plt.xlabel('X - axis')
plt.ylabel('Y - axis')
plt.title('Not linearly separable')

clf = Perceptron()
clf.fit(phi_x,y)
#w0, w1 = clf.coef_[0]

resolution = 0.01
x1_min, x1_max = X[:, 0].min(), X[:, 0].max() 
x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
np.arange(x2_min, x2_max, resolution))
K = np.array([xx1.ravel(), xx2.ravel()]).T
test_feat = feature_mapping(K[:,0],K[:,1],10)
Z = clf.predict(test_feat)
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.4)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())

plt.legend(loc='upper right')
plt.show()