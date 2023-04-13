#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 00:32:06 2021

@author: mohamad
"""

from train import Train
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt

class Test:
    def __init__(self, *args, **kwargs):
        self.test=Train()
        pass
    
    def test_func(self,test_f):
        x_test = []
        y_test = []
        model_train = load_model("model.h5")
        for features,label in test_f:
            x_test.append(features)
            y_test.append(label)
        x_test = np.array(x_test) / 255
        x_test = x_test.reshape(-1, 150, 150, 1)
        y_test = np.array(y_test)
        print("Loss of the model is - " , model_train.evaluate(x_test,y_test)[0])
        print("Accuracy of the model is - " , model_train.evaluate(x_test,y_test)[1]) 
        predictions = model_train.predict_classes(x_test)
        predictions = predictions.reshape(1,-1)[0]
        
        return predictions ,y_test ,x_test
    def report(self,predictions ,y_test):
        cm = confusion_matrix(y_test,predictions)
        return cm ,classification_report(y_test, predictions, 
                    target_names = ['Pneumonia (Class 0)','Normal (Class 1)'])
   
    
    
    def disolay(self,predictions ,y_test,x_test):
        correct = np.nonzero(predictions == y_test)[0]
        incorrect = np.nonzero(predictions != y_test)[0] 
      

        i = 0
        for c in correct[:6]:
            plt.subplot(3,2,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x_test[c].reshape(150,150), cmap="gray", interpolation='none')
            plt.title("Predicted Class {},Actual Class {}".format(predictions[c], y_test[c]))
            plt.tight_layout()
            i += 1



if __name__=='__main__':
    
   
    print('djskjdk')
    # predictions ,y_test=model.test_func(test_f)
    # cm, str1=model.report(predictions, y_test)
    # print(cm)
    
    