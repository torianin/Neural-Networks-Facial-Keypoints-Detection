#!/usr/bin/env python

import numpy as np
import pandas
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

from utils import load_model, store_model, filter_data, load_data

import pylearn2
from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer

CUDA = True

def train_network(X, y):
    if CUDA:
        Conv2DLayer = layers.cuda_convnet.Conv2DCCLayer
        MaxPool2DLayer = layers.cuda_convnet.MaxPool2DCCLayer
    else:
        Conv2DLayer = layers.Conv2DLayer
        MaxPool2DLayer = layers.MaxPool2DLayer
    
    net2 = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', Conv2DLayer),
            ('pool1', MaxPool2DLayer),
            ('conv2', Conv2DLayer),
            ('pool2', MaxPool2DLayer),
            ('conv3', Conv2DLayer),
            ('pool3', MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('dropout', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 1, 96, 96),
        conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
        conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2),
        conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_ds=(2, 2),
        hidden4_num_units=500, hidden5_num_units=500,
        dropout_p=0.5,
        output_num_units=30, output_nonlinearity=None,
    
        update_learning_rate=0.01,
        update_momentum=0.9,
    
        regression=True,
        max_epochs=2000,
        verbose=1,
        )
    net2.fit(X, y)
    return net2

X, y = load_data()
X, y = filter_data(X, y, range(30))
net = train_network(X, y)
#store_model([X, y], 'data.pickle')
store_model(net, 'main_net.pickle')
