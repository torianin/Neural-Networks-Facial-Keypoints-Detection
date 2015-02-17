#!/usr/bin/env python

import numpy as np
import pandas
import argparse
from utils import load_model, store_model, predict, feature_idx, load_specialists

def load_data():
    data = pandas.io.parsers.read_csv("test.csv")
    data['Image'] = data['Image'].apply(lambda x: np.array(x.split(), dtype=np.float32))
    X = np.vstack(data['Image'].values) / 255.
    return X.reshape(-1, 1, 96, 96)

def print_submission(predictions):
    lookup = pandas.io.parsers.read_csv("IdLookupTable.csv")
    print "RowId,Location"
    for idx, row in enumerate(lookup.values):
        image_idx = int(row[1])-1
        answer = predictions[image_idx][feature_idx(row[2])]
        print "%s,%lf"%(row[0], min(answer, 96))

parser = argparse.ArgumentParser()
parser.add_argument("net", help="pickle file with trained net")
args = parser.parse_args()

X = load_data()
net = load_model(args.net)
specialists = load_specialists()

predictions = predict(net, specialists, X)

print_submission(predictions)
