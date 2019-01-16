# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 22:46:15 2018

@author: diaae
"""
import numpy as np
from functions import *
import matplotlib.pyplot as plt


class ANN (object):
    def __init__(self):
        self.layer = {}
        self.parameters = {}
        
 
    def add(self,dim,activation = None): 
        layer_number = len(self.layer) 
#        assert (layer_number  == 0 and activation != None)
        self.layer['L'+str(layer_number)] = {'dim':dim,'act':activation}
            
                
    def fit(self,X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost=False):

        costs = []                         # keep track of cost
        input_size = self.layer['L0']['dim']
        output_size = self.layer['L'+str(len(self.layer)-1)]['dim']
        print(self.layer)
        Y = Y.reshape((len(Y),output_size))
        X = X.reshape((len(X),input_size))
        # Parameters initialization. (≈ 1 line of code)
        ### START CODE HERE ###
        self.parameters = initialize_parameters_deep(self.layer)
        ### END CODE HERE ###
        
        # Loop (gradient descent)
        for i in range(0, num_iterations):
    
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            ### START CODE HERE ### (≈ 1 line of code)

           
            for t in range(len(X)):
                x = X[t]
                y = Y[t]
                x = x.reshape((input_size,1))
                y = y.reshape((output_size,1))
#                print(x.shape,y.shape)
                AL, caches = L_model_forward(x, self.parameters)
    
                ### END CODE HERE ###
                
                # Compute cost.
                ### START CODE HERE ### (≈ 1 line of code)
                cost = compute_cost(AL, y)
                ### END CODE HERE ###
            
                # Backward propagation.
                ### START CODE HERE ### (≈ 1 line of code)
                grads = L_model_backward(AL, y, caches,self.layer)
                ### END CODE HERE ###
         
                # Update parameters.
                ### START CODE HERE ### (≈ 1 line of code)
                self.parameters = update_parameters(self.parameters, grads, learning_rate)
                ### END CODE HERE ###
                    
            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        return self.parameters
    
    
    def predict(self,X):
        X = X.reshape((len(X),1))
        prediction = L_model_forward(X, self.parameters,)
        return prediction
    
