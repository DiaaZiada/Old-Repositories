from keras.preprocessing.image import img_to_array
import imutils
import cv2
from os import listdir
import os
import numpy as np


class Data(object):
    def __init__(self):
        self.t = []
        self.images = []
        self.describe = {'Expressions':[],'Illumination':[],'Pose':[],'Occlusion':[],'Age':[],'Makeup':[],'Gender':[]}
        self.dd = {
    'Expressions': [
        'ANGER',
        'DISGUST',
        'FEAR',
        'HAPPINESS',
        'NEUTRAL',
        'SADNESS',
        'SURPRISE',
    ],
    'Illumination': [
        'BAD',
        'HIGH',
        'MEDIUM',
    ],
    'Pose': [
        'DOWN',
        'FRONTAL',
        'LEFT',
        'RIGHT',
        'UP',
    ],
    'Occlusion': [
        'BEARD',
        'GLASSES',
        'HAIR',
        'HAND',
        'NONE',
        'ORNAMENTS',
        'OTHERS',
    ],
    'Age': [
        'CHILD',
        'MIDDLE',
        'OLD',
        'YOUNG',
    ],
    'Makeup': [
        'OVER',
        'PARTIAL',
    ],
    'Gender': [
        'FEMALE',
        'MALE',
    ]
}

    def collect_data(self, path):

        if not os.path.isdir(path):
            return
        if 'images' in os.listdir(path):
            self.__collector(path)
            return
        for dir in listdir(path):
            self.collect_data(str(path + '\\' + dir))
        return (self.images, self.describe)

    def __collector(self, path):
        x = [x for x in listdir(path) if '.txt' in x]
        file_path = path + '\\' + x[0]
        with open(file_path, 'r') as file:
            try:
                data = file.readlines()
                for f in data[1:]:
                    f = f.split('\t')[:-1]
                    f = [x.strip() for x in f]

                    image_name = [n for n in f if '.jpg' in n]
                    image_path = path + '\\images\\' + image_name[-1]

                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.resize(image,(28,28))
                    image = img_to_array(image)
                    self.images.append(image)
                    self.t.append(f[2])
                    self.describe['Makeup'].append([n for n in f if n in self.dd['Makeup']])
                    self.describe['Pose'].append([n for n in f if n in self.dd['Pose']][-1])
                    self.describe['Age'].append([n for n in f if n in self.dd['Age']])
                    self.describe['Illumination'].append([n for n in f if n in self.dd['Illumination']])
                    self.describe['Occlusion'].append([n for n in f if n in self.dd['Occlusion']])
                    self.describe['Expressions'].append([n for n in f if n in self.dd['Expressions']])
                    self.describe['Gender'].append([n for n in f if n in self.dd['Gender']])
                    if len(self.describe['Age']) != len(self.images):
                        print(self.describe['Age'][-2])
                        print(len(self.describe['Age']),len(self.images))
                        print(image_path)
                        break
            except:
                pass


# d = Data()
# d.collect_data(r'E:\faces\IMFDB_final')
# o = np.array(d.t)
# print(np.unique(np.array(d.describe['Pose'])))
# print(len(np.array(d.describe['Pose'])))
# print(len(np.array(d.describe['Age'])))
# print(len(np.array(d.describe['Expressions'])))
# print(len(np.array(d.describe['Illumination'])))
# print(len(d.images))
