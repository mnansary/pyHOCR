# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

from keras.models import load_model

class PostProcessor(object):
    def __init__(self,model_path=None):
        if model_path:
            self.model=load_model(model_path)
    
    def test_model(self):
        st="ভাল"
        print(st)

        