# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 19:40:58 2018

@author: diaae
"""

import cv2
import numpy as np
import imutils
import scipy
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input


model = load_model('model.hdf5')




camera = cv2.VideoCapture(0)


# keep looping
while True:

    # grab the current frame
    (grabbed, frame) = camera.read()
    frame = imutils.resize(frame, width=500)
    hand = frame[50:250, 50:250]
    hand2 = hand

    hand = scipy.misc.imresize(hand, size=(64,64))
    hand = image.img_to_array(hand)
    hand = np.expand_dims(hand, axis=0)
    hand = preprocess_input(hand)


    result = model.predict(hand)
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

