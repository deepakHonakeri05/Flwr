import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import cv2

import os

import numpy as np
from keras.preprocessing import image


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout,Conv2D,MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(vertical_flip=True,
                                 horizontal_flip=True,
                                 rotation_range=180,
                                 zoom_range=0.2)

test_set = test_datagen.flow_from_directory(
        'test/',
        target_size=(224, 224),
        batch_size=32,
        shuffle=True,
        #seed=101,
        class_mode='categorical')

model = tf.keras.models.load_model("models/model_client_1_epoch10.h5")


scores = model.evaluate_generator(generator=test_set)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
