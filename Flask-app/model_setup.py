#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf 
import numpy as np 
import os 

import sys
sys.path.append("..") # Adds higher directory to python modules path

from DenseNet.models import denseNet

class Deploy(object):
    def __init__(self):
        model,_=denseNet()
        model.load_weights('DenseNet.h5')
        self.symbol_list=[
                    'অ','আ','ই','ঈ','উ',
                    'ঊ','ঋ','এ','ঐ','ও',
                    'ঔ','ক','খ','গ','ঘ',
                    'ঙ','চ','ছ','জ','ঝ',
                    'ঞ','ট','ঠ','ড','ঢ',
                    'ণ','ত','থ','দ','ধ',
                    'ন','প','ফ','ব','ভ',
                    'ম','য','র','ল','শ',
                    'ষ','স','হ','ড়','ঢ়',
                    'য়','ৎ','ং','ঃ','ঁ'
                    ] 
        self.model=model
        self.graph= tf.get_default_graph()

    def predict_symbol(self,img_path):
        img_data=load_img(img_path,color_mode = "grayscale",target_size=(32,32))
        arr=img_to_array(img_data)
        arr=arr.astype('float32')/255
        pred = np.argmax(self.model.predict(np.expand_dims(arr,axis=0)))
        print(colored('The predicted Symbol:{}'.format(self.symbol_list[pred]),'green'))
        return self.symbol_list[pred]

if __name__=='__main__':
    img_path='data.png'
    DeployObj=Deploy()
    res=DeployObj.predict_symbol(img_path)
    print(res)