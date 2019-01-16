# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 19:40:58 2018

@author: diaae
"""

import cv2
import numpy as np
import imutils
import tensorflow as tf
import sys
from tf_utils import predict
import scipy
parameters = np.load('model.npy')

#
## Retrieve the parameters from the dictionary "parameters" 
#
#W1 = tf.convert_to_tensor(parameters.item().get('W1'), np.float32)
#b1 = tf.convert_to_tensor(parameters.item().get('b1'), np.float32)
#W2 = tf.convert_to_tensor(parameters.item().get('W2'), np.float32)
#b2 = tf.convert_to_tensor(parameters.item().get('b2'), np.float32)
#W3 = tf.convert_to_tensor(parameters.item().get('W3'), np.float32)
#b3 = tf.convert_to_tensor(parameters.item().get('b3'), np.float32)



W1 = tf.convert_to_tensor(parameters.item().get('W1'))
b1 = tf.convert_to_tensor(parameters.item().get('b1'))
W2 = tf.convert_to_tensor(parameters.item().get('W2'))
b2 = tf.convert_to_tensor(parameters.item().get('b2'))
W3 = tf.convert_to_tensor(parameters.item().get('W3'))
b3 = tf.convert_to_tensor(parameters.item().get('b3'))
    
params = {"W1": W1,
          "b1": b1,
           "W2": W2,
           "b2": b2,
           "W3": W3,
           "b3": b3}

camera = cv2.VideoCapture(0)






# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    

    
    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = np.add(np.multiply(W1, X), b1)                                             # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    Z2 = np.add(np.multiply(W2, A1), b2)                                              # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
    Z3 = np.add(np.multiply(W3, A2), b3)                                              # Z3 = np.dot(W3,Z2) + b3
    ### END CODE HERE ###
    
    return Z3




# keep looping
while True:

    # grab the current frame
    (grabbed, frame) = camera.read()
    frame = imutils.resize(frame, width=500)
    hand = frame[50:250, 50:250]
    hand2 = hand

    hand = scipy.misc.imresize(hand, size=(64,64)).reshape((1, 64*64*3)).T

    result = predict(hand, params)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    cv2.putText(hand2,str(result),(0,20), font, 1,(255,255,255),5) 
    cv2.imshow("hand", hand2)

    print(result)
    
    # if the ’q’ key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

