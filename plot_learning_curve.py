#!/usr/bin/env python

from utils import load_model
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("net", help="pickle file with trained net")
args = parser.parse_args()

net = load_model(args.net)
train_loss = [x['train_loss'] for x in net.train_history_]
valid_loss = [x['valid_loss'] for x in net.train_history_]

plt.xlabel('epochs')
plt.yscale('log')
plt.plot(train_loss, label="training loss")
plt.plot(valid_loss, label="validation loss")
plt.legend()
fig = plt.figure(1)
fig.savefig('learning_curve.png')
