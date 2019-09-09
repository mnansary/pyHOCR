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

import h5py

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


def saveh5(data,iden):
    hf = h5py.File('{}.h5'.format(iden) , 'w')
    hf.create_dataset('data',data=data)
    hf.close()
    print('Saved: \t {}.h5'.format(iden))

def info_preprocess(dset_dir):
    dset=DataSet(dset_dir)
    Xt,Yt,Xv,Yv=dset.preprocess()
    Xtt=dset.test_tensors
    Ytt=dset.test_classes
    saveh5(Xt,'Xt')
    saveh5(Yt,'Yt')
    saveh5(Xv,'Xv')
    saveh5(Yv,'Yv')
    saveh5(Xtt,'Xtt')
    saveh5(Ytt,'Ytt')

def readh5(d_path):
    data=h5py.File(d_path, 'r')
    data = np.array(data['data'])
    return data

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='info script utils.py')
    parser.add_argument("dset_dir", help="/path/to/Data/Folder")
    args = parser.parse_args()
    dset_dir=args.dset_dir
    info_preprocess(dset_dir)
    Xt=readh5('Xt.h5')
    Yt=readh5('Yt.h5')
    Xv=readh5('Xv.h5')
    Yv=readh5('Yv.h5')
    Xtt=readh5('Xtt.h5')
    Ytt=readh5('Ytt.h5')
    print(Xt.shape)
    print(Yt.shape)
    print(Xv.shape)
    print(Yv.shape)
    print(Xtt.shape)
    print(Ytt.shape)
    