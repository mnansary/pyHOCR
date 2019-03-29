import os
import numpy as np 
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical 
from sklearn.datasets import load_files 
from sklearn.model_selection import train_test_split
class Preprocessor(object):
    def __init__(self,database_path=None,resize_dim=(32,32),test_data_identifier='Test',train_data_identifier='Train'):
        self.__database_path=database_path
        self.__resize_dim=resize_dim
        self.__test_data_path=os.path.join(self.__database_path,test_data_identifier)
        self.__train_data_path=os.path.join(self.__database_path,train_data_identifier)
    
    def __load_dataset(self,path):
        print('#    Loading Data from: {}'.format(path))
        data_bunch=load_files(path)
        file_names=np.array(data_bunch['filenames'])
        targets=np.array(data_bunch['target'])
        classes=to_categorical(targets,num_classes=50)
        return file_names,classes

    def __validation_test_train_data(self):
        print('#    Extracting Test Train Validation data')
        self.__test_files,self.__test_classes=self.__load_dataset(self.__test_data_path)
        self.__model_files,self.__model_classes=self.__load_dataset(self.__train_data_path)
        self.__train_files,self.__validation_files,self.train_classes,self.validation_classes=train_test_split(self.__model_files,self.__model_classes,test_size=0.2,stratify=self.__model_classes)
    
    
    def __convert_file_to_tensor(self,file_name,binarize=True):
        img=load_img(file_name,color_mode = "grayscale",target_size=self.__resize_dim)
        arr=img_to_array(img)
        if binarize:
            arr[arr>0]=2
            arr[arr==0]=1
            arr[arr==2]=0
        tensor=np.expand_dims(arr,axis=0)
        return tensor

    def __files_to_tensors(self,file_names):
        list_of_tensors = [self.__convert_file_to_tensor(file_name) for file_name in file_names]
        tensors=np.vstack(list_of_tensors)
        return tensors

    def __load_tensors(self):
        print('#    Loading Tensors')
        self.train_tensors = self.__files_to_tensors(self.__train_files)
        self.validation_tensors = self.__files_to_tensors(self.__validation_files)
        self.test_tensors = self.__files_to_tensors(self.__test_files)
    
    def preprocess_data(self):
        print('#    Preprocessing Data')
        self.__validation_test_train_data()
        self.__load_tensors()
        return [self.train_tensors,self.train_classes,self.validation_tensors,self.validation_classes]
        

        
        

