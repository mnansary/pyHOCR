# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import img_to_array 
from keras.models import load_model
from preprocessing.utils import Preprocessor
from sklearn import metrics

class PostProcessor(object):
    def __init__(self,model_path):
        
        self.model=load_model(model_path)
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
                        ]                        # 'ঁ' not print able
        
    def test_model(self,database_path):
        preprocObj=Preprocessor(database_path=database_path)
        preprocObj.preprocess_data(return_flag=False)    	       
        test_tensors=preprocObj.test_tensors	   
        test_classes=preprocObj.test_classes
        
        print(colored('# Generating Predictions','blue'))
        predictions = [np.argmax(self.model.predict(np.expand_dims(tensor,axis=0))) for tensor in test_tensors]
        
        print(colored('# Getting Ground Truth','blue'))	    
        ground_truth = [np.argmax(truth_value) for truth_value in test_classes]
        
        print(colored('# Calculating Accuracy','blue'))	    
        
        prediction_accuracy = 100* metrics.f1_score(ground_truth,predictions, average = 'micro')	   
        print(colored('Test data Prediction Accuracy [F1 accuracy]: {}'.format(prediction_accuracy),'green'))
    

    def predict_symbol(self,img_path,plot_flag=False,resize_dim=(64,64),app_data=False):
        img_data = cv2.imread(img_path,0)
        
        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(img_data,(5,5),0)
        
        _,thresholded_data = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        img_data = cv2.resize(thresholded_data,resize_dim) 
        
        
        if plot_flag:
            plt.title('Thresholded {}'.format(img_path))
            plt.imshow(thresholded_data)
            plt.show()
            plt.clf()
            plt.close()
            
        
        tensor=img_to_array(img_data)

        tensor=tensor.astype('float32')/255
        
        pred = np.argmax(self.model.predict(np.expand_dims(tensor,axis=0)))
        
        print(colored("The predicted symbol is : ","blue")+colored(self.symbol_list[pred],"green")) 

        if app_data:
            return self.symbol_list[pred]