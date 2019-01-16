# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras.backend as K
from keras.utils import np_utils
from collections import Counter
from conv.lenet import LeNet
from utils.data import Data
import numpy as np
import matplotlib.pyplot as plt

# init variables
data_path = r'E:\faces\IMFDB_final'
model_name = 'Occlusion'
output_model_path = r'E:\faces\output_model'

# data preparing class
data = Data()

# load the image, pre-process it, and store it in the data list
print('collect data ')
data.collect_data(data_path)
images = data.images
describe = data.describe

# scale the raw pixel intensities to the range [0, 1]
images = np.array(images, dtype="float") / 255.0

# take only requerment for model\
describe = describe[model_name]

# number of classes
no_classes = len(np.unique(describe))

# convert the labels from integers to vectors
describe = np.array(describe).flatten()

le = LabelEncoder().fit(describe)
describe = np_utils.to_categorical(le.transform(describe), no_classes)

# ignore math errors
np.seterr(divide='ignore', invalid='ignore')

# account for skew in the labeled data
classTotals = describe.sum(axis=0)
classWeight = classTotals.max() / classTotals

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(images, describe, test_size=0.20, stratify=describe, random_state=42)

# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=1, classes=no_classes)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), class_weight=classWeight, batch_size=64, epochs=15, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model_save = output_model_path + '\\' + model_name + '.hdf5'
model.save(model_save)
print("done")
