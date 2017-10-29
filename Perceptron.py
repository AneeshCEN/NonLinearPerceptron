# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 11:18:38 2017

@author: ANEESH
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 12:31:13 2017

@author: ANEESH
"""

import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.linear_model import Perceptron
np.random.seed(6)


class Perceptron():
    
    def __init__(self, training_data, label):
        self.X = training_data
        self.y = label
        #Learning rate
        self.eta = 0.01
        #number of iteration 
        self.num_iter = 500
        
        self.weights = np.zeros(self.X.shape[1])
        
        self.UpdateWeights()
        

#
    def HypothesisPrediction(self, x):
        prediction = np.dot(self.weights, x)
        if prediction>=0:
            return 1
        else:
            return -1

    def WeightUpdation(self):
        for x,y in zip(self.X,self.y):
            predicted_label = self.HypothesisPrediction(x)
            error = self.eta*(y-predicted_label)
            delta_w =  error *x
            self.weights = self.weights+delta_w
        

    def UpdateWeights(self):
        for i in range(self.num_iter):
            self.WeightUpdation()

           


if __name__ == "__main__":
    
    #number of iteration 
    num_iter = 500
   
   
    eta = 0.01
    
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    plt.scatter(X[:50,0],X[:50,1], c='r',label='class 1 (Labels -1)')
    plt.scatter(X[50:100,0],X[50:100,1], c='b',label='class 2 (Labels +1)')
    plt.title('Linearly separable')
    plt.xlabel('X - axis')
    plt.ylabel('Y - axis')
    plt.legend(loc='upper right')
#    plt.show()
    
    training_data = X[:100,:]
    
    # Appending feature vector of ones to hold for bias term
    X = np.c_[np.ones(training_data.shape[0]), training_data]
    y = y[:100,]
    
    # Just changing all the 0 lables into -1
    y[np.where(y==0)] = -1
    
    # initialize the weight vector
#    weights = np.zeros(X.shape[1])
#                
#    # update weights in each iteration  
#    for i in range(num_iter):
#        weights = update_weights(weights, X, y)
#        weights = weights
        
    # Using scikit learn library 
#    clf = Perceptron()
#    clf.fit(X,y)
#    w0,w1,w2 = clf.coef_[0]
    
    
    x_min = min(X[:100,1])
    x_max = max(X[:100,1])
    
    y_min = min(X[:100,2])
    y_max = max(X[:100,2])
    
    
    x_axis = np.linspace(x_min,x_max)
    y_axis = np.linspace(y_min,y_max)
    
    clf = Perceptron(X, y)
    weights = clf.weights
        
    Y = -(weights[0]+weights[1]*x_axis)/weights[2]
#    Y2 = -(w0+w1*x_axis)/w2
    plt.plot(x_axis,Y,c='black')
#    plt.plot(x_axis,Y2, c='g')
    plt.ylim(y_min-1,y_max+1)
    plt.legend(loc='best')
    plt.show()

    
    