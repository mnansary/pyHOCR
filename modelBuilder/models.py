# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

from keras.models import Model
from keras.layers import Activation, Convolution2D, Dropout, GlobalAveragePooling2D
from keras.layers import Concatenate, Dense, Input, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

class DenseNet(object):
    def __init__(
        self,image_dim=(64,64,1),
        num_of_classes=50,
        num_of_layers=6,
        num_of_dense_block=3,
        growth_rate=16, 
        num_of_filter=16, 
        dropout_rate=None,
        weight_decay=1E-4,
        compression=1.0,
        add_bottleneck=False):
        """
        Arguments:
            image_dim= shape of image input -> Model Input
            num_of_classes= number of classer -> Output of Dense Layer with activation
            num_of_layers= number of convolution blocks -> Conv layers inside Dense Blocks
            num_of_dense_block= number of Dense blocks
            growth_rate= increase in number of filters 
            droupout_rate and weight_decay are self explanatory :) 
            bottleneck=adding 1x1 convolution in block 
            compression=reduce the number of feature-maps at transition layer
        """
        self.model_input = Input(shape=image_dim)
        self.num_of_classes=num_of_classes
        self.num_of_layers = num_of_layers
        self.num_of_dense_block=num_of_dense_block
        self.num_of_filter=num_of_filter
        self.growth_rate=growth_rate
        self.num_of_dense_block=num_of_dense_block
        self.dropout_rate=dropout_rate
        self.weight_decay=weight_decay
        self.compression=compression
        self.add_bottleneck=add_bottleneck
        self.feature_count=self.num_of_filter

        if self.compression <=0.0 or self.compression > 1.0:
            raise Exception('Compression have to be a value between 0.0 and 1.0.')
        
        if add_bottleneck:
            self.num_of_layers = int(self.num_of_layers/2)

        self.num_of_layers=[self.num_of_layers for _ in range(self.num_of_dense_block)]

    def __initialConvLayer(self):
        """
        Initial Conv2D with BatchNormalization
        """
        self.output= Convolution2D(self.num_of_filter*2,(3,3), padding='same',strides=(1,1),
                      use_bias=False, kernel_regularizer=l2(self.weight_decay))(self.model_input)
        
        self.output = BatchNormalization()(self.output)

    def __transitionBlock(self):
        """
            BatchNorm>Activate>Convolute>Dropout>pool
        """
        self.output = BatchNormalization()(self.output)
        self.output = Activation('relu')(self.output)
        self.output = Convolution2D(int(self.feature_count*self.compression), (1, 1), padding='same',
                      use_bias=False, kernel_regularizer=l2(self.weight_decay))(self.output)
        if self.dropout_rate:
            self.output = Dropout(self.dropout_rate)(self.output)
        
        self.output = AveragePooling2D((2, 2), strides=(2, 2))(self.output)
    
    def __convBlock(self):
        """
            Bottleneck>BatchNorm>Activate>Convolute>Dropout
        """
        if self.add_bottleneck:
            bottleneck_width = 4
            self.output = BatchNormalization()(self.output)
            self.output = Activation('relu')(self.output)
            self.output = Convolution2D(self.feature_count * bottleneck_width, (1, 1), use_bias=False, kernel_regularizer=l2(self.weight_decay))(self.output)
            if self.dropout_rate:
                self.output = Dropout(self.dropout_rate)(self.output)

        self.output = BatchNormalization()(self.output)
        self.output = Activation('relu')(self.output)
        self.output = Convolution2D(self.feature_count, (3, 3), padding='same', use_bias=False)(self.output)
        if self.dropout_rate:
            self.output = Dropout(self.dropout_rate)(self.output)

    def __denseBlock(self,layer_count):
        """
        convblock |> concatenate |> update feature count
        """
        outputs=[self.output]
        for _ in range(layer_count):
            self.__convBlock()
            outputs.append(self.output)
            self.output = Concatenate(axis=-1)(outputs)
            self.feature_count += self.growth_rate
    
    def __fullyConnect(self):
        """
        BatchNorm >Activate>GlobalPool>Dense(FCL)
        """
        self.output = BatchNormalization()(self.output)
        self.output = Activation('relu')(self.output)
        self.output = GlobalAveragePooling2D()(self.output)
        self.output = Dense(self.num_of_classes, activation='softmax')(self.output)

    def buildDenseNet(self):
        """
        Conv2d > Dense Blocks |> Transitions|>LDB>Fully Connected   
        """
        if self.dropout_rate:
            dropout_flag=self.dropout_rate
        else:
            dropout_flag='none'
        
        if self.add_bottleneck:
            bottleneck_flag='yes'
        else:
            bottleneck_flag='no'
        

        self.model_name='DenseNet-Class:{}-Shape:{}-dropout:{}-bottleneck:{} '.format(self.num_of_classes,self.model_input,dropout_flag,bottleneck_flag)

        self.__initialConvLayer()

        for layer_num in range(self.num_of_dense_block - 1):
            self.__denseBlock(self.num_of_layers[layer_num])
            self.__transitionBlock()
            self.feature_count = int(self.feature_count * self.compression)
        
        self.__denseBlock(self.num_of_layers[-1])
        self.__fullyConnect()
        
        self.DenseNetModel=Model(input=self.model_input,output=self.output,name=self.model_name)