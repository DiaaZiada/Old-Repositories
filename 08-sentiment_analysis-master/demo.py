# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 18:12:19 2018

@author: diaae
"""
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

print('Loading data...')
(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=20000)
print('data downloaded')

x_train = sequence.pad_sequences(x_train,maxlen=80)
x_test = sequence.pad_sequences(x_test,maxlen=80)

model = Sequential()
model.add(Embedding(20000,128))
model.add(LSTM(128,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train,y_train,
          batch_size=32,
          epochs=15,
          verbose=2,
          validation_data=(x_test,y_test))


score, acc = model.evaluate(x_test,y_test,
                            batch_size=32,
                            verbose=2)
print('Test score:',score)
print('test accuracy:',acc)

#model.save('sentimentanalysis.hdf5')


