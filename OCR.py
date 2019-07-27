#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

from flask import Flask,render_template,request,url_for
from binascii import a2b_base64

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

import matplotlib.pyplot as plt 

import numpy as np 

import os 

img_path=os.path.join(os.getcwd(),'test_images','data.png')

model_path='/home/ansary/WORK/OCR/Models/MODEL.hdf5'

model=load_model(model_path)
print(colored('#        Model Loaded ','green'))
app=Flask(__name__)

symbol_list=[
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
            ] # 'ঁ' not print able

# Helper Functions
def saveImg(data):
    image_data=str(data).split(',')[1]
    binary_data = a2b_base64(image_data)
    with open(img_path,'wb') as data:
        data.write(binary_data)

def readImg(img_path):
    img=load_img(img_path,color_mode = "grayscale",target_size=(64,64))
    tensor=img_to_array(img)
    tensor=tensor.astype('float32')/255    
    pred = np.argmax(model.predict(np.expand_dims(tensor,axis=0)))
    return pred

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction
@app.route('/predict/',methods=['GET','POST'])

def predict():
    string_res='Data Saved'

    try:
        data=request.get_data()
    except Exception as e: 
        print(colored("#        Can Not Receieve Data",'red'))
        print(e)
        string_res='FAILED TO FETCH DATA'
    
    try:
        saveImg(data)
    except Exception as e: 
        print(colored("#        Can Not Save Image",'red'))
        print(e)
        string_res='FAILED TO SAVE DATA'
    
    pred=readImg(img_path)
    
    string_res=symbol_list[pred]
    
    return string_res




if __name__=='__main__':
    # App in Debug Mode    
    app.run(debug=True)
