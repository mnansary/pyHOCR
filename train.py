# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import argparse
from preprocessing.utils import Preprocessor 
from trainingModels.densenet import DenseNet
from keras.callbacks import ModelCheckpoint

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

def train_data(data_shape,model_name,data,epochs=250,batch_size=32):
    model=DenseNet(input_shape=data_shape,nb_classes=50,dense_blocks=1,depth=15,dropout_rate=0.2,compression=0.5)
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

if __name__ == "__main__":
    data=preprocess(database_path)    
    train_data(data_shape,model_name,data)