import flwr as fl
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout,Conv2D,MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np

def getModel():
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
    #pooling layer
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
    return model

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
        '/Users/deepak/utd/2sem/cv/projects/2 project/flowers_dataset/train2/',
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

custom_model = getModel()
log_filename='log2.txt'

# Define Flower client
class CifarClient(fl.client.NumPyClient):
  def get_parameters(self):
    return custom_model.get_weights()

  def fit(self, parameters, config):
#    print(parameters)
    custom_model.set_weights(parameters)
    os.system('rm -rf model_client2.h5')
    history_logger=tf.keras.callbacks.CSVLogger(log_filename, separator=",", append=True)
    history = custom_model.fit_generator( train_set, epochs=1,callbacks=[history_logger], validation_data=test_set, validation_steps=10)
    np.save('history2.npy',history.history)
    
    custom_model.save("./model_client2.h5")
    return custom_model.get_weights(), train_set.samples, {}

  def evaluate(self, parameters, config):
    model1 = getModel()
    model1.set_weights(parameters)
    loss, accuracy = model1.evaluate_generator(generator=test_set)
    return loss, test_set.samples, {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client("[::]:8080", client=CifarClient())

#plt.figure(1, figsize = (15,8))
#    
#plt.subplot(221)
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.title('accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'valid'])
#    
#plt.subplot(222)
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'valid'])
#
#plt.show()
