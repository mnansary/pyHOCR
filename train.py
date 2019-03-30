# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import argparse
import numpy as np 
from preprocessing.utils import Preprocessor 
from trainingModels.densenet import DenseNet
from keras.callbacks import ModelCheckpoint
from keras import Sequential
from sklearn import metrics

parser = argparse.ArgumentParser(description='Bangla OCR 50 class Alphabet DenseNet Model')
parser.add_argument("datafolder", help="/path/to/Data/Folder")

args = parser.parse_args()
database_path=args.datafolder

model_name='denseNet.hdf5'

data_shape=(32,32,1)

def preprocess(database_path):
    preprocObj=Preprocessor(database_path=database_path)
    data=preprocObj.preprocess_data()    
    return data

def train_data(data_shape,model_name,data,epochs=250,batch_size=100):
    model=DenseNet(input_shape=data_shape,nb_classes=50)
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    epochs = epochs
    batch_size = batch_size
    checkpoint = ModelCheckpoint(filepath='saved_models/{}'.format(model_name), verbose=1, save_best_only=True)
    train_tensors=data[0]
    train_classes=data[1]
    validation_tensors=data[2]
    validation_classes=data[3]
    model.fit(train_tensors, train_classes,validation_data=(validation_tensors, validation_classes),epochs=epochs, batch_size=batch_size, callbacks=[checkpoint], verbose=1)

def test_data(model_name):
    preprocObj=Preprocessor(database_path=database_path)
    preprocObj.preprocess_data(return_flag=False)
    
    test_tensors=preprocObj.test_tensors
    test_targets=preprocObj.test_classes

    model=DenseNet(input_shape=data_shape,nb_classes=50)
    model.load_weights('saved_models/{}'.format(model_name))
    
    alphabet_predictions = [np.argmax(model.predict(np.expand_dims(tensor,axis=0))) for tensor in test_tensors]
    
    y_true = [np.argmax(y_test) for y_test in test_targets]
    
    f1_accuracy = 100* metrics.f1_score(y_true,alphabet_predictions, average = 'micro')
    
    print('Test F1 accuracy: %.4f%%' % f1_accuracy)

if __name__ == "__main__":
    data=preprocess(database_path)    
    #train_data(data_shape,model_name,data)
    test_data(model_name)