#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:57:03 2021

@author: mohamad
"""




import numpy as np
import pandas as pd
import os
import cv2
import random
import matplotlib.pyplot as plt
import seaborn as sns


from random import randint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras.utils import np_utils
from keras.models import Sequential
from keras import optimizers
from keras import backend as K
from keras.layers import Dense, Activation, Flatten, Dense,MaxPool2D, Dropout
from keras.layers import Conv2D, MaxPool2D, BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

data = []


class Train:
    def __init__(self, *args, **kwargs):
     
        pass
    def adjust_gamma(self,image, gamma = 1.0):
        invGamma = 1.0 / gamma
    
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
        return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))
    
    def sampel_image(self):
        
        im=[]
        new_dir = 'chest_xray/train'

        for image in os.listdir(new_dir):
            
            image_folder = os.path.join(new_dir, image)
            rnd_number = randint(0, len(os.listdir(image_folder)))
            image_file = os.listdir(image_folder)[rnd_number]
            image_file = os.path.join(image_folder, image_file)
            image_show = cv2.imread(image_file)
            im.append(image_file)
        return im
        
    def get_data(self,directory):
        gamma = 2.5
        normal_count = 0
        pneumonia_count = 0
        img_size = 150
        assign_dict = {"NORMAL":0, "PNEUMONIA":1}
        for sub_directory in os.listdir(directory):
            if sub_directory == "NORMAL":
                inner_directory = os.path.join(directory,sub_directory)
                for i in os.listdir(inner_directory):
                    print(directory+':'+i)
                    img = cv2.imread(os.path.join(inner_directory,i),0)
                    img =  self.adjust_gamma(img, gamma=gamma)
                    img = cv2.resize(img,(img_size,img_size))
                    data.append([img,assign_dict[sub_directory]])
                    
            if sub_directory == "PNEUMONIA":
                inner_directory = os.path.join(directory,sub_directory)
                for i in os.listdir(inner_directory):
                    print(directory+':'+i)
                    img = cv2.imread(os.path.join(inner_directory,i),0)
                    img =  self.adjust_gamma(img, gamma=gamma)
                    img = cv2.resize(img,(img_size,img_size))
                    data.append([img,assign_dict[sub_directory]])
        random.shuffle(data)  
        return np.array(data)

    def display_bar(self,data):
          l = []
          for i in train:
              if(i[1] == 0):
                  l.append("Normal")
              else:
                  l.append("Pneumonia")
          sns.set_style('darkgrid')
          sns.countplot(l)
    def data_image(self,data):
       
        Pneumonia=0
        Normal=0
        total_image=0
    
        for category in data:
            total_image+=1
            if category[1]==1:
                Pneumonia+=1
            else:
                Normal+=1
        return total_image ,Normal,Pneumonia
    
    def func_train(self,train,val):
        
        x_train = []
        y_train = []
        for features,label in train:
              x_train.append(features)
              y_train.append(label)


        x_value = []
        y_value = []
        for features,label in val:
            x_value.append(features)
            y_value.append(label)

       # print(x_train)
        x_train = np.array(x_train) / 255.
        x_val = np.array(x_value) / 255
       
        img_size = 150

    # resize data for deep learning 
        x_train = x_train.reshape(-1, img_size, img_size, 1)
        y_train = np.array(y_train)

        x_val = x_val.reshape(-1, img_size, img_size, 1)
        y_val = np.array(y_value)

       
        
        model = Sequential()
        model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (150,150,1)))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(Flatten())
        model.add(Dense(units = 128 , activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units = 1 , activation = 'sigmoid'))
        model.compile(optimizer = "rmsprop" , loss = 'binary_crossentropy' , metrics = ['accuracy'])
        model.summary()
        # reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', 
        #                       patience = 2, 
        #                       verbose=1,
        #                       factor=0.3, 
        #                       min_lr=0.000001)
        history = model.fit(x_train,y_train, batch_size = 32,epochs = 10 ,validation_data=(x_val,y_val))
                                  
        return history ,model
        


