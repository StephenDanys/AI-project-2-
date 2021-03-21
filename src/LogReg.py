# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 17:02:09 2021

@author: pchrk
"""
import numpy as np

class LogisticRegression():
    
    def __init__ (self,lr=0.001, n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None
        
    #we will try to change the hyperplane position through fiddling with the weights
    def fit(self,x,y):
        print("started training with Logistic Regression")
        n_samples, n_features=x.shape
        self.weights=np.zeros(n_features)
        self.bias=0
        
        for _ in range(self.n_iters):
            
            #line-decision boundary
            line_decision_boundary=np.dot(x,self.weights)+self.bias
            yhat=self.sigmoid_func(line_decision_boundary)
            
            dw=(1/n_samples)*np.dot((x.T),(yhat-y))
            db=(1/n_samples)*np.sum(yhat-y)
            
            self.weights-=self.lr*dw
            self.bias-=self.lr*db
        
    
    def predict(self, x):
        #x is a vector 
        #ldb in d-weight space
        line_decision_boundary=np.dot(x, self.weights)+self.bias
        #squeeze through the sigmoid func to get a number between [0-1]
        probabilistic_yhat=self.sigmoid_func(line_decision_boundary)
        yhat=[1 if i>0.5 else 0 for i in probabilistic_yhat]
        return yhat
    
    def sigmoid_func(self,x):
        return 1/(1+np.exp(-x))
    
"""
class LOGREG2():
    def __init__ (self,lr=0.001, n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None
    
    def sigmoid_func(self,x):
        return 1/(1+np.exp(-x)) 
    
    def sigmoid_tonos(self,x):
        return np.log(x)*(1-np.log(x))
    
    def log_log_likelihood_i(self,x_i,y_i,bias):
        if y_i==1:
            return np.log(sigmoid_func(np.dot(x_i,bias)))
        else:
            return np.log(1-sigmoid_func(np.dot(x_i,beta)))
        
    def log_log_likelihood(self,x,y,bias):
        return np.sum(log_log_likelihood_i(x_i,y_i,bias)
                      for x_i,y_i in zip(x,y))"""