#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

from flask import Flask,render_template,request,url_for
from binascii import a2b_base64
from model_setup import Deploy

DeployObj=Deploy()
    
app=Flask(__name__)
# Helper Functions
img_path='data.png'
def saveImg(data):
    image_data=str(data).split(',')[1]
    binary_data = a2b_base64(image_data)
    with open(img_path,'wb') as data:
        data.write(binary_data)


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
        print(colored("# Can Not Receieve Data",'red'))
        print(e)
        string_res='FAILED TO FETCH DATA'
    
    print(colored("# Receieved Data",'red'))

    try:
        saveImg(data)
    except Exception as e: 
        print(colored("#Can Not Save Image",'red'))
        print(e)
        string_res='FAILED TO SAVE DATA'
    
    print(colored("# Saved Data",'red'))
    
    with DeployObj.graph.as_default():
        string_res=DeployObj.predict_symbol(img_path)
    
    return string_res




if __name__=='__main__':
    # App in Debug Mode    
    app.run(debug=True)
