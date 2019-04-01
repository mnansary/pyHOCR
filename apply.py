# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import argparse
import numpy as np 
from preprocessing.utils import Preprocessor 
from trainingModels.densenet import DenseNet

parser = argparse.ArgumentParser(description='Bangla OCR 50 class Alphabet DenseNet Model')
parser.add_argument("model_path", help="/path/to/model")
parser.add_argument("img_file", help="/path/to/image/file")

args = parser.parse_args()
model_path=args.model_path
img_file=args.img_file

data_shape=(32,32,1)


## LIMITED BY TECH

def preprocess_img(img_file):
    pass

def load_model(model_path):
    print('# Loading Models')
    model=DenseNet(input_shape=data_shape,nb_classes=50)
    model.load_weights(model_path)


    
