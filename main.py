#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 20:33:30 2021

@author: mohamad
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:52:54 2020

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

from train import Train
from test_widghet import Test_widghet
class Mlp(QMainWindow):
    
    def __init__(self):
        
        QMainWindow.__init__(self)

        loadUi("dising_main.ui",self)
        self.model=Train()
        im=self.model.sampel_image()
        self.showMaximized()
        print(im[0])
        self.pixmap_image = QtGui.QPixmap(im[0])
        self.label.setPixmap(self.pixmap_image)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setScaledContents(True)
        self.label.setMinimumSize(1,1)
        self.label.show()
        pixmap_imag1 = QtGui.QPixmap(im[1])
        self.label_2.setPixmap(pixmap_imag1)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setScaledContents(True)
        self.label_2.setMinimumSize(1,1)
        self.label_2.show()
        self.p_train.clicked.connect(self.f_train)
        self.p_test.clicked.connect(self.f_test)
        self.p_train.pressed.connect(self.f_total)
    
    
        
    
    
    
    def f_test(self):
        self.ui=Test_widghet();
        self.ui.show()
    def f_train(self):
        train = self.model.get_data('../chest_xray/train')
        val = self.model.get_data('../chest_xray/val')
        history ,model_train=self.model.func_train(train, val)
        model_train.save('model.h5')
       #
        epochs = [i for i in range(10)]
        
        train_acc = history.history['accuracy']
        train_loss = history.history['loss']
        val_acc = history.history['val_accuracy']
        val_loss = history.history['val_loss']
       
    
        self.widget.canvas.axes.plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
        self.widget.canvas.axes.plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
        self.widget.canvas.axes.set_title('Training & Validation Accuracy')
        self.widget.canvas.axes.legend()
        self.widget.canvas.axes.set_xlabel("Epochs")
        self.widget.canvas.axes.set_ylabel("Accuracy")
        
        self.widget.canvas.axes1.plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
        self.widget.canvas.axes1.plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
        self.widget.canvas.axes1.set_title('Training & Validation & Loss')
        self.widget.canvas.axes1.legend()
        self.widget.canvas.axes1.set_xlabel("Epochs")
        self.widget.canvas.axes1.set_ylabel("Training & Validation Loss")
        self.widget.canvas.draw()
        stringlist = []
        model_train.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        self.consol.append(short_model_summary)
       
    def f_total(self):
        
        self.label_total.setText('5216')
        self.label_norm.setText('1341')
        self.label_p.setText('3875')
        self.consol.append("clicked Train Waitin ....")




if __name__=="__main__":
    
    app = QApplication([])
    window = Mlp()
    window.show()
    app.exec_()