#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

from modelUseage.useage import PostProcessor
import argparse

parser = argparse.ArgumentParser(description='Bangla OCR 50 class Alphabet DenseNet Model -- Testing Script')
parser.add_argument("datafolder", help="/path/to/Data/Folder")
parser.add_argument("model_path",help="/path/to/test/Model.hdf5")

args = parser.parse_args()
database_path=args.datafolder
model_path=args.model_path

def test(model_path,database_path):
    obj=PostProcessor(model_path)
    obj.test_model(database_path)

if __name__=='__main__':
    test(model_path,database_path)