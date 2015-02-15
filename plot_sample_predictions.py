#!/usr/bin/env python

import numpy as np
import pandas
import argparse
from utils import load_model, store_model

import matplotlib.pyplot as pyplot

def load_data():
    data = pandas.io.parsers.read_csv("test.csv")
    data['Image'] = data['Image'].apply(lambda x: np.array(x.split(), dtype=np.float32))
    X = np.vstack(data['Image'].values) / 255.
    return X

def plot_samples(net, X, outfile):
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
    
    fig = pyplot.figure(1)
    fig.savefig(outfile)

parser = argparse.ArgumentParser()
parser.add_argument("net", help="pickle file with trained net")
parser.add_argument("outfile", help="output .png file")
args = parser.parse_args()

X = load_data()
net = load_model(args.net)
plot_samples(net, X[np.random.permutation(len(X))][:20], args.outfile)
