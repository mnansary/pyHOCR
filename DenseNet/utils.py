# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

import os
import numpy as np 

from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical 

from sklearn.datasets import load_files 
from sklearn.model_selection import train_test_split

def image_to_tensor(file_name,resize_dim=(64,64)):
    img=load_img(file_name,color_mode = "grayscale",target_size=resize_dim)
    arr=img_to_array(img)
    tensor=np.expand_dims(arr,axis=0)
    return tensor


class DataSet(object):
    def __init__(self,dset_dir,resize_dim=(64,64)):
        '''
            Alphabet Final DataSet Specific
        '''
        self.dset_dir=dset_dir
        self.resize_dim=resize_dim
        self.test_dir=os.path.join(dset_dir,'Test')
        self.train_dir=os.path.join(dset_dir,'Train')
        self.nb_class=50

    def __load(self,path):
        print(colored('#    Loading Data from: {}'.format(path),'blue'))
        data_bunch=load_files(path)
        file_names=np.array(data_bunch['filenames'])
        targets=np.array(data_bunch['target'])
        classes=to_categorical(targets,num_classes=self.nb_class)
        return file_names,classes,targets

    def __files_to_tensors(self,file_names):
        list_of_tensors = [image_to_tensor(file_name,resize_dim=self.resize_dim) for file_name in file_names]
        tensors=np.vstack(list_of_tensors)
        return tensors
    
    def __split(self):
        print(colored('#    Extracting Test Train Validation data','blue'))
        self.test_files,self.test_classes,self.test_targets=self.__load(self.test_dir)
        self.model_files,self.model_classes,self.model_targets=self.__load(self.train_dir)
        self.train_files,self.valid_files,self.train_classes,self.valid_classes=train_test_split(self.model_files,self.model_classes,test_size=0.2,stratify=self.model_classes)
    

    def __load_tensors(self):
        print('#    Loading Tensors')
        self.train_tensors = self.__files_to_tensors(self.train_files).astype('float32')/255
        self.valid_tensors = self.__files_to_tensors(self.valid_files).astype('float32')/255
        self.test_tensors = self.__files_to_tensors(self.test_files).astype('float32')/255
    
    def preprocess(self,return_flag=True):
        print('#    Preprocessing Data')
        self.__split()
        self.__load_tensors()
        if return_flag:
            return [self.train_tensors,self.train_classes,self.valid_tensors,self.valid_classes]


def info_preprocess(dset_dir):
    dset=DataSet(dset_dir)
    trt,_,vlt,_=dset.preprocess()
    print(trt.shape)
    print(vlt.shape)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='info script utils.py')
    parser.add_argument("dset_dir", help="/path/to/Data/Folder")
    args = parser.parse_args()
    dset_dir=args.dset_dir
    info_preprocess(dset_dir)