# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import scipy.signal
from scipy.ndimage import rotate
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.interpolation import shift
from skimage.morphology import skeletonize_3d 
import os 

from matplotlib import gridspec


class Separator(object):
    def __init__(self,img_file_path):
        self.img_file_path=img_file_path
        self.output_dir=None
    
    def __binarizeData(self): 
        print(colored('# Binarizing Image!!','green'))
        img_data = cv2.imread(self.img_file_path,0)
        self.plotData(img_data,identifier='Raw Data')
        blur = cv2.GaussianBlur(img_data,(5,5),0)
        _,thresholded_data = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)    
        thresholded_data[thresholded_data<255]=0
        thresholded_data[thresholded_data==255]=1
        self.data=1-thresholded_data
       
    
    def __mapConnectedComponents(self):
        print(colored('# Tracking Connected Components !!','green'))
        kernel = np.ones((5,5), np.uint8) 
        dilated_data = cv2.dilate(self.data, kernel, iterations=6)     
        labeled_data,num_of_components =scipy.ndimage.measurements.label(dilated_data)
        diff=self.data*num_of_components - labeled_data
        diff[diff<0]=0
        self.plotData(diff,identifier='Detected Words/Segments')
        print(colored('# Storing Words !!','green'))
        self.words=[]
        self.sk_words=[]
        self.row_sum=[]
        self.col_sum=[]
        for component in range(1,num_of_components):
            idx = np.where(diff==component)
            y,h,x,w = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
            word=np.ones((h-y+1,w-x+1))    
            idx=(idx[0]-y,idx[1]-x)
            word[idx]=0
            inv_word=1-word
            sk_word=skeletonize_3d(inv_word)/255
            
            #rotate
            coords = np.column_stack(np.where(sk_word > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            # rotate the image to deskew it
            (h, w) = inv_word.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(sk_word, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            self.row_sum.append(np.sum(1-rotated,axis=1))
            self.col_sum.append(np.sum(1-rotated,axis=0))
            self.sk_words.append(rotated)
            self.words.append(word)

        self.words.reverse()
        self.sk_words.reverse()

    def __nullifySingle(self,line_width=1):

        for i in range(len(self.words)):

            word=self.words[i]
            self.plotData(word)
            max_y=np.nanmax(self.col_sum[i])

            thresh_y=(max_y-line_width)/max_y
            
            y=self.col_sum[i]/max_y
            
            cols=np.argwhere(y>=thresh_y)
            
            for col in cols:
                word[:,col]=1
            self.plotData(word)
            



    def preprocessData(self,img_file_path,return_img=False):
        self.img_file_path=img_file_path
        self.__binarizeData()
        self.plotData(self.data,identifier="Binary Data")        
        self.__mapConnectedComponents()
        self.debug_plothist()
        #self.__nullifySingle()
        if return_img:
            return self.data

    
    def plotData(self,data,plot_now_flag=True,save_plot_flag=False,identifier=None):
        if identifier:
            plt.figure(identifier)
            plt.title(identifier)
        
        plt.grid(True)

        plt.imshow(data)

        if plot_now_flag:
            plt.show()
        
        if save_plot_flag:
            print('Saving {} at {}'.format(identifier+'.png',self.output_dir))
            plt.savefig(os.path.join(self.output_dir,identifier+'.png'))
        
        plt.clf()
        plt.close()

    def debug_plothist(self,plot_now_flag=True,save_plot_flag=False):
        for i in range(len(self.words)):
            fig = plt.figure('Word/Segment {}'.format(i))
            
            gs = gridspec.GridSpec(2, 2,width_ratios=[10,1],height_ratios=[5,1]) 
            
            img_ax = plt.subplot(gs[0])
            img_ax.imshow(self.words[i]+self.sk_words[i])

            x=self.row_sum[i]/np.nanmax(self.row_sum[i])
            y=np.flip(np.arange(x.shape[0]))
            col_hist = plt.subplot(gs[1])
            col_hist.plot(x,y)
            
            
            y=self.col_sum[i]/np.nanmax(self.col_sum[i])
            x=np.arange(y.shape[0])
            row_hist = plt.subplot(gs[2])
            row_hist.plot(x,y)
        
            if plot_now_flag:
                plt.show()

            if save_plot_flag:
                print('Saving {} at {}'.format(identifier+'.png',self.output_dir))
                plt.savefig(os.path.join(self.output_dir,identifier+'.png'))
            
            plt.clf()
            plt.close()

if __name__=='__main__':
    img_file_path='test.jpg'
    SeparatorObj=Separator(img_file_path)
    SeparatorObj.preprocessData(img_file_path)
