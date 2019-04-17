#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

import argparse
import numpy as np 
from preprocessing.utils import Preprocessor 
from modelBuilder.models import DenseNet
from keras.callbacks import ModelCheckpoint,TensorBoard
from sklearn import metrics
import os

parser = argparse.ArgumentParser(description='Bangla OCR 50 class Alphabet DenseNet Model -- Training Script')
parser.add_argument("datafolder", help="/path/to/Data/Folder")

args = parser.parse_args()
database_path=args.datafolder


def preProcess(database_path):
    preprocObj=Preprocessor(database_path=database_path)
    data=preprocObj.preprocess_data()    
    return data

def trainModel(data,epochs=250,batch_size=30,optimizer_func='rmsprop'):
    
    modelObj=DenseNet()
    modelObj.buildDenseNet()

    model=modelObj.DenseNetModel

    model_name=modelObj.model_name+'optimizer:{}.hdf5'.format(optimizer_func)

    print(colored('Training model:{}'.format(model_name),'red'))
    
    model.summary()

    model.compile(optimizer=optimizer_func, loss='categorical_crossentropy', metrics=['accuracy'])
    epochs = epochs
    batch_size = batch_size
    
    checkpoint = ModelCheckpoint(filepath='saved_models/{}'.format(model_name), verbose=1, save_best_only=True)
    
    train_tensors=data[0]
    train_classes=data[1]
    
    validation_tensors=data[2]
    validation_classes=data[3]
    
    tensorboard=TensorBoard(os.path.join("saved_models","LOGS",model_name))
    history=model.fit(train_tensors, train_classes,validation_data=(validation_tensors, validation_classes),epochs=epochs, batch_size=batch_size, callbacks=[checkpoint,tensorboard], verbose=1)
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(os.path.join('saved_models/',model_name+'_loss'+'.png'))
    plt.clf()
    plt.close()    
    model.save(os.path.join('saved_models','FINAL_'+model_name))

    
if __name__ == "__main__":
    data=preProcess(database_path)    
    trainModel(data)