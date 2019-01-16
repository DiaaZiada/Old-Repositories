
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 21:33:13 2018

@author: diaae
"""

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import pandas_datareader.data as web


start = dt.datetime(2012, 1, 1)
end = dt.datetime(2017,1,1)

dataset = web.DataReader('GOOGL', 'morningstar', start, end)
training_set = dataset.iloc[:,3:4].values


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 20 timesteps and t+1 output
X_train = []
y_train = []
for i in range(20, 1258):
    X_train.append(training_set_scaled[i-20:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
model = Sequential()

# Adding the input layer and the LSTM layer
model.add(LSTM(units = 3, input_shape = (None, 1)))

# Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN
model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 100, batch_size = 32)


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price for February 1st 2012 - January 31st 2017

#
start = dt.datetime(2017, 1, 1)
end = dt.datetime.now()
#
dataset = web.DataReader('GOOGL', 'morningstar', start, end)
test_set = dataset.iloc[:,3:4].values
test_set = np.array([test_set[i] for i in range(0,len(test_set),30)])


real_stock_price = np.concatenate((training_set, test_set), axis = 0)

# Getting the predicted stock price of 2017
scaled_real_stock_price = sc.fit_transform(real_stock_price)
inputs = []
for i in range(len(training_set), len(real_stock_price)):
    inputs.append(scaled_real_stock_price[i-20:i, 0])
inputs = np.array(inputs)
inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
predicted_stock_price = model.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price[len(training_set):], color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
#
#
## Save keras model
#model.save('keras_model')
#
## Save tensorflowjs model
#import tensorflowjs as tfjs
#tfjs.converters.save_keras_model(model,'tfjs_model')