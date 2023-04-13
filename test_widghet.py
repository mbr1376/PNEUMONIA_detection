#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 00:20:59 2021

@author: mohamad
"""



from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import random
import time    
from random import randint
import matplotlib.image as mpimg
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report,confusion_matrix
from train import Train
from test_p import Test
from numpy import asarray

class Test_widghet(QMainWindow):
    
    def __init__(self):
        
        QMainWindow.__init__(self)
        self.temp=0
        loadUi("test.ui",self)
        self.back_button.clicked.connect(self.back)
        self.import_img.clicked.connect(self.read)
        self.import_folder.clicked.connect(self.read_folder)
        self.run_test.clicked.connect(self.test_function)

    
        
    
    def test_function(self):
        if self.temp==1:
             model=Test()
             test_f = model.test.get_data(self.dir_path)
             predictions ,y_test,x_test=model.test_func(test_f)
             cm, str1=model.report(predictions, y_test)
             self.consol.append(str1)
             res = [[str(ele) for ele in sub] for sub in cm] 
             
             self.consol.append("confusion_matrix:"+str(res))
             model.disolay(predictions, y_test, x_test)
        if self.temp==2:
            print('fdjjkdfjkfd')
            img = cv2.imread(self.path,0)
            m_train = load_model("model.h5")
            img = np.array(img)
            img = img.astype("float32") / 255.
            img=cv2.resize(img,(150,150))
            img = img.reshape( img.shape[0] ,img.shape[1],1)
            yhat = m_train.predict(asarray([img]))
            print(yhat)
            if yhat[0]==1:
               self.consol.append("image is PNEUMONIA") 
            else:
                self.consol.append("image is Normal") 
            
    def read(self):
        self.path=fname = QFileDialog.getOpenFileName(None, "Window name", "", "image files (*.*)")[0]

        print(self.path)
        self.pixmap_image = QtGui.QPixmap(self.path)
        self.label.setPixmap(self.pixmap_image)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setScaledContents(True)
        self.label.setMinimumSize(1,1)
        self.label.show()
        self.consol.append("readed images")
        self.temp=2
    def  read_folder(self):
         self.dir_path=QFileDialog.getExistingDirectory(None,"Choose Directory","E:\\",QFileDialog.ShowDirsOnly)
         self.consol.append("readed  folder data")
         self.temp=1
         
    def back(self):
        self.close()


if __name__=="__main__":
    
    app = QApplication([])
    window = Test_widghet()
    window.show()
    app.exec_()