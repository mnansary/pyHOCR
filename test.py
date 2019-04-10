#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

from postprocessing.modelLoader import PostProcessor

def test():
    obj=PostProcessor()
    obj.test_model()

if __name__=='__main__':
    test()