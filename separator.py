# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

import argparse

parser = argparse.ArgumentParser(description="Separation Of Bangla Symbols From Images")

parser.add_argument("img_file_path", help="/path/to/img/file")

args = parser.parse_args()

img_file_path=args.img_file_path

import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import os 
import scipy.signal
from scipy.misc import imsave
from PIL import Image

class Separator(object):
    def __init__(self):
        self.img_file_path=None
        self.output_dir=None
    
    def __binarizeData(self): 
        print(colored('# Binarizing Image!!','green'))
        img_data = cv2.imread(self.img_file_path,0)
        self.plotData(img_data,identifier='Raw Data',save_plot_flag=False)
        blur = cv2.GaussianBlur(img_data,(5,5),0)
        _,thresholded_data = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)    
        thresholded_data[thresholded_data<255]=0
        thresholded_data[thresholded_data==255]=1
        self.data=1-thresholded_data
       
    
    def __mapConnectedComponents(self):
        print(colored('# Tracking Connected Components !!','green'))
        kernel = np.ones((5,5), np.uint8) 
        dilated_data = cv2.dilate(self.data, kernel, iterations=8)     
        labeled_data,num_of_components =scipy.ndimage.measurements.label(dilated_data)
        diff=self.data*num_of_components - labeled_data
        
        diff[diff<0]=0
        print(colored('# Storing symbols !!','green'))
        
        self.symbols=[]
        for component in range(1,num_of_components):
            idx = np.where(diff==component)
            y,h,x,w = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
            symbol=np.ones((h-y+1,w-x+1))    
            idx=(idx[0]-y,idx[1]-x)
            symbol[idx]=0
            symbol=np.array(Image.fromarray(symbol).resize((64,64))) 
            self.plotData(symbol,identifier='{}'.format(component),plot_now_flag=False)
            self.symbols.append(symbol)
        

    def preprocessData(self,img_file_path):
        self.img_file_path=img_file_path
        self.__binarizeData()
        self.plotData(self.data,identifier="Binary Data",plot_now_flag=True,save_plot_flag=False)        
        self.__mapConnectedComponents()
       

    
    def plotData(self,data,plot_now_flag=True,save_plot_flag=True,identifier=None):
        if identifier:
            plt.figure(identifier)
            plt.title(identifier)
        
        plt.grid(True)

        plt.imshow(data)

        if plot_now_flag:
            plt.show()
        
        if save_plot_flag:
            print('Saving {} at {}'.format(identifier+'.png',self.output_dir))
            imsave(os.path.join(self.output_dir,identifier+'.png'),data)

        plt.clf()
        plt.close()

if __name__ == "__main__":
    preprocObj=Separator()
    preprocObj.output_dir='./test_images'
    preprocObj.preprocessData(img_file_path)
    