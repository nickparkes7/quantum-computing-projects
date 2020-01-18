# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 19:29:46 2019

@author: nickp
"""

import numpy as np
from scipy.special import expit
from scipy.io import loadmat


class NeuralNetwork:
    """
    Classical Neural Network representation with one inner layer of an
    arbitrary height
    """
    
    def __init__(self, data_in_size, inner_layer_size, out_size, learn_rate):
        self.weights1 = np.random.rand(data_in_size, inner_layer_size)
        self.weights2 = np.random.rand(inner_layer_size, out_size)
        self.learn_rate = learn_rate
        self.output = np.zeros(out_size)
        
        
    def run(self, data_in):
        self.feedforward(data_in)
        return self.output
        
    
    def train(self, data_in, expected):
        self.feedforward(data_in)
        self.backprop(data_in, expected)
       
        
    def feedforward(self, data_in):
        self.layer1 = expit(np.dot(data_in, self.weights1))
        self.output = expit(np.dot(self.layer1, self.weights2))
        
        
    def backprop(self, data_in, expected):
        z2 = np.dot(self.layer1, self.weights2)
        z1 = np.dot(data_in, self.weights1)
        
        # compute differentials for weights and biases
        d_weights2 = np.dot(self.layer1.reshape(self.layer1.size,1), 
                            (2 * (expected - self.output) * expit(z2) * 
                            (1 - expit(z2))).reshape(1,expected.size))
        d_weights1 = np.dot(data_in.reshape(data_in.size,1), 
                            (np.dot(2 * (expected - self.output) * expit(z2) * 
                            (1 - expit(z2)), self.weights2.T) * expit(z1) *
                            (1 - expit(z1))).reshape(1, self.layer1.size))
        #d_bias2 = np.dot(2*(self.y - self.output), (expit(z2)*(1-expit(z2))).T)
        #d_bias1 = np.dot(np.dot(2 * (self.y - self.output) * expit(z2) * 
        #                    (1 - expit(z2)), self.weights2.T), (expit(z1) *
        #                    (1 - expit(z1))).T)
        
        # update weights and biases
        self.weights2 = self.weights2 + self.learn_rate * d_weights2
        self.weights1 = self.weights1 + self.learn_rate * d_weights1
        #self.bias2 = self.bias2 + self.learn_rate * d_bias2
        #self.bias1 = self.bias1 + self.learn_rate * d_bias1
        

if __name__ == "__main__":
    
    """
    nn = NeuralNetwork(2, 6, 2, 0.0001)
    
    for i in range(1500):
        x = np.random.uniform(0,2)
        y = np.random.uniform(0,4)
        exp = np.array([1,0])
        nn.train(np.array([x,y]), exp)
        
    for i in range(1500):
        x = np.random.uniform(2,4)
        y = np.random.uniform(0,4)
        exp = np.array([0,1])
        nn.train(np.array([x,y]), exp)
        
    
    for i in range(10):
        x = np.random.uniform(0,4)
        y = np.random.uniform(0,4)
        print("x =", x)
        print("y =", y)
        soln = nn.run(np.array([x,y]))
        print("prob a =", soln[0])
        print("prob b =", soln[1])
        print("")
    """
    nn = NeuralNetwork(400, 1000, 10, 0.001)
    
    dataset = loadmat("ex3data1.mat")
    X = dataset['X']
    y = dataset['y']
    
    r = np.arange(0, 5000)
    np.random.shuffle(r)
    
    for i in np.arange(0, 4500):
        x_train = X[r[i],:]
        y_train = np.zeros(10)
        y_train[y[r[i]] - 1] = 1
        nn.train(x_train, y_train)
    
    for i in np.arange(4500, 5000, 1):
        x_run = X[r[i],:]
        print("expected =", y[r[i]])
        print("result = ", nn.run(x_run))
        print("")
    
    
    
    
    
    