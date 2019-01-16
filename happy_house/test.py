# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 02:31:33 2018

@author: diaae
"""

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input

import numpy as np
import argparse
import imutils
import cv2
from keras.preprocessing import image

# path for haar cascade
cascade_path = 'haarcascade_frontalface_alt2.xml'

# path for video
video_path = ''

video_bool = video_path != ''


# load each model
model = load_model('happyModel.hdf5')



# load the face detector cascade and smile detector CNN
detector = cv2.CascadeClassifier(cascade_path)

# check is a video or camera
if video_bool:
    camera = cv2.VideoCapture(video_path)
else:
    camera = cv2.VideoCapture(0)

# keep looping
while True:

    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a frame, then we
    # have reached the end of the video
    if video_bool and not grabbed:
        break

    # resize the frame, convert it to grayscale, and then clone the
    # original frame so we can draw on it later in the program
    frame = imutils.resize(frame, width=500)
    
    frameClone = frame.copy()

    # detect faces in the input frame, then clone the frame so that
    # we can draw on it
    rects = detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # loop over the face bounding boxes
    for (fX, fY, fW, fH) in rects:

        # extract the ROI of the face from the grayscale image,
        # resize it to a fixed 28x28 pixels, and then prepare the
        # ROI for classification via the CNN
        roi = frame[fY:fY + fH, fX:fX + fW]
        # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (64, 64))
        
        
#        roi = image.img_to_array(roi)
#        roi = np.expand_dims(roi, axis=0)
#        roi = preprocess_input(roi)
        
        roi = img_to_array(roi)
        roi = roi.astype("float") / 255.0
        roi = np.expand_dims(roi, axis=0)

        # detect models predictions
        (pred) = model.predict(roi)
        print(pred)

        if pred[0][0] > 0.5:
            pred = 'happy'
        else:
            pred = 'sad'
        label = pred 
        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frameClone, label, (fX + fW + 10, fY + 25 ), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)
        # show our detected faces along with all models
    cv2.imshow("Face", frameClone)
    # if the ’q’ key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
