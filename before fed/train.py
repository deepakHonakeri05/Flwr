# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# %matplotlib inline 

import cv2

import os

import numpy as np
from keras.preprocessing import image

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout,Conv2D,MaxPooling2D
from tensorflow.keras.layers import BatchNormalization

model=Sequential()
#covolution layer
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)))
#pooling layer
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
    #covolution layer
model.add(Conv2D(32,(3,3),activation='relu'))
    #pooling layer
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
    #covolution layer
model.add(Conv2D(64,(3,3),activation='relu'))
    #pooling laye
    
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
    #covolution layer
model.add(Conv2D(64,(3,3),activation='relu'))
    #pooling layer
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
    #i/p layer
model.add(Flatten())
    #o/p layer
model.add(Dense(13,activation='softmax'))
    # Load model and data (MobileNetV2, CIFAR-10)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range = 270,
    # brightness_range = (-2,2),
    #     shear_range=0.2,
    #     zoom_range=0.3,
    vertical_flip=True,
    validation_split=0.3,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(vertical_flip=True,
                                 horizontal_flip=True,
                                 rotation_range=180,
                                 zoom_range=0.2)

train_set = train_datagen.flow_from_directory(
        '/Users/deepak/utd/2sem/cv/projects/2 project/flowers_dataset/train1/',
        target_size=(224,224),
        batch_size=32,
        #seed=101, 
        shuffle=True,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        '/Users/deepak/utd/2sem/cv/projects/2 project/flowers_dataset/test/',
        target_size=(224, 224),
        batch_size=32,
        shuffle=True,
        #seed=101,
        class_mode='categorical')

print(train_set.class_indices)

history = model.fit_generator(
        train_set,
        epochs=10,
        validation_data=test_set,
        validation_steps=10)

model.save("./model_client1.h5")
#model.save_weights("./classifier_weights_resnet_attempt3.h5")

scores = model.evaluate_generator(generator=test_set)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

plt.figure(1, figsize = (15,8)) 
    
plt.subplot(221)  
plt.plot(history.history['accuracy'])  
plt.plot(history.history['val_accuracy'])  
plt.title('accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 
    
plt.subplot(222)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 

plt.show()

thresh = 0.90
test_image = image.load_img('./test_4.jpg',target_size=(224,224))  
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = model.predict(test_image)
#print(result)
classes = ['astilbe', 'bellflower', 'black-eyed susan', 'calendula', 'california poppy','carnation', 'common daisy', 'coreopsis', 'dandelion', 'iris', 'rose', 'sunflower', 'tulip']

for i,x in enumerate(result[0]):
  if x >= thresh:
    print(classes[i], x)
