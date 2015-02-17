#!/usr/bin/env python

import numpy as np
import pandas
from sklearn.base import clone
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator

from utils import load_model, store_model, feature_idx, load_data, filter_data
from specialist_settings import SPECIALIST_SETTINGS

import pylearn2
from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer

CUDA = True

def get_network():
    if CUDA:
        Conv2DLayer = layers.cuda_convnet.Conv2DCCLayer
        MaxPool2DLayer = layers.cuda_convnet.MaxPool2DCCLayer
    else:
        Conv2DLayer = layers.Conv2DLayer
        MaxPool2DLayer = layers.MaxPool2DLayer
    
    net = NeuralNet(
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
        max_epochs=500,
        verbose=1,
        )
    return net

def train_specialist(main_net, columns, X, y):
    net = get_network()
    net.output_num_units = len(columns)
    net.max_epochs = 400
    net.load_weights_from(main_net)
    net.fit(X,y)
    return net

X, y = load_data()
main_net = load_model("main_net.pickle")
nets = []
for s in SPECIALIST_SETTINGS:
    cols = map(feature_idx, s['columns'])
    sX, sy = filter_data(X, y,cols)
    print s['name']
    print cols
    print sX.shape
    print sy.shape
    net = train_specialist(main_net, cols, X, sy)
    nets.append(net)
    store_model(nets, "specialist_"+s['name']+".pickle")
