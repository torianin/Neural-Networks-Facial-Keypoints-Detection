#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

import numpy as np
import pandas
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

from utils import load_model, store_model

pyplot = plt

def show_plot():
    fig = plt.figure(1)
    fig.set_size_inches(10,10)
    fig.savefig("figure.png")
    os.system("eog figure.png")

def show_img(img):
    plt.imshow(img.reshape((96,96)), cmap = cm.Greys_r) #.set_interpolation('nearest')
    show_plot()

def load_data():
    #data = open("training.csv", "r")
    data = pandas.io.parsers.read_csv("training.csv")
    data['Image'] = data['Image'].apply(lambda x: np.array(x.split(), dtype=np.float32))
    data = data.dropna()
    X = np.vstack(data['Image'].values) / 255.
    y = data[data.columns[:-1]].values
    y = (y - 48) / 48
    y = y.astype(np.float32)
    p = np.random.permutation(len(X))
    X = X[p]
    y = y[p]
    return X, y


def train_network(X, y):
    net1 = NeuralNet(
        layers=[  # three layers: one hidden layer
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        # layer parameters:
        input_shape=(None, 9216),  # 96x96 input pixels per batch
        hidden_num_units=100,  # number of units in hidden layer
        output_nonlinearity=None,  # output layer uses identity function
        output_num_units=30,  # 30 target values
    
        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,
    
        regression=True,  # flag to indicate we're dealing with regression problem
        max_epochs=400,  # we want to train this many epochs
        verbose=1,
        )
    net1.fit(X, y)
    return net1

def train_network2(X, y):
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
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 1, 96, 96),
        conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
        conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2),
        conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_ds=(2, 2),
        hidden4_num_units=500, hidden5_num_units=500,
        output_num_units=30, output_nonlinearity=None,
    
        update_learning_rate=0.01,
        update_momentum=0.9,
    
        regression=True,
        max_epochs=10,
        verbose=1,
        )
    return net2


# use the cuda-convnet implementations of conv and max-pool layer
#Conv2DLayer = layers.cuda_convnet.Conv2DCCLayer
#MaxPool2DLayer = layers.cuda_convnet.MaxPool2DCCLayer


#X, y = load_data()
#store_model([X, y], 'data.pickle')
#X, y = load2d()  # load 2-d data
#net2.fit(X.reshape(-1, 1, 96, 96), y)
#store_model(net2, 'conv_net.pickle')

X, y = load_model('data.pickle')
net2 = load_model('conv_net.pickle')

#testX = X[:100]
#X = X[100:]
#y = y[100:]
#net = train_network(X, y)
#store_model(net, 'simple_net.pickle')
#net1 = load_model('simple_net.pickle')
def plot_samples(net, X):
    def plot_sample(x, y, axis):
        img = x.reshape(96, 96)
        axis.imshow(img, cmap='gray')
        axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)
    
    y_pred = net.predict(X)
    
    fig = pyplot.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], y_pred[i], ax)
    
    show_plot()

X = X[:100].reshape(-1, 1, 96, 96)
plot_samples(net2, X)
