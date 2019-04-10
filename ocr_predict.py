# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

from modelUseage.useage import PostProcessor
import argparse

parser = argparse.ArgumentParser(description='Bangla OCR 50 class Alphabet DenseNet Model -- Predictor Script')
parser.add_argument("img_path", help="/path/to/img/file")
parser.add_argument("model_path",help="/path/to/test/Model.hdf5")

args = parser.parse_args()
img_path=args.img_path
model_path=args.model_path

def symbol_test(model_path,img_path):
    obj=PostProcessor(model_path)
    obj.predict_symbol(img_path,plot_flag=True)

if __name__=='__main__':
    symbol_test(model_path,img_path)