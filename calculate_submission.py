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

def calculate_submission(net, X):
    features = "left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y,left_eye_inner_corner_x,left_eye_inner_corner_y,left_eye_outer_corner_x,left_eye_outer_corner_y,right_eye_inner_corner_x,right_eye_inner_corner_y,right_eye_outer_corner_x,right_eye_outer_corner_y,left_eyebrow_inner_end_x,left_eyebrow_inner_end_y,left_eyebrow_outer_end_x,left_eyebrow_outer_end_y,right_eyebrow_inner_end_x,right_eyebrow_inner_end_y,right_eyebrow_outer_end_x,right_eyebrow_outer_end_y,nose_tip_x,nose_tip_y,mouth_left_corner_x,mouth_left_corner_y,mouth_right_corner_x,mouth_right_corner_y,mouth_center_top_lip_x,mouth_center_top_lip_y,mouth_center_bottom_lip_x,mouth_center_bottom_lip_y".split(",")
    lookup = pandas.io.parsers.read_csv("IdLookupTable.csv")
    predictions = net.predict(X)*48+48
    print "RowId,Location"
    for idx, row in enumerate(lookup.values):
        image_idx = int(row[1])-1
        feature_idx = features.index(row[2])
        answer = predictions[image_idx][feature_idx]
        print "%s,%lf"%(row[0], min(answer, 96))

parser = argparse.ArgumentParser()
parser.add_argument("net", help="pickle file with trained net")
args = parser.parse_args()

X = load_data()
X = X.reshape(-1, 1, 96, 96)
net = load_model(args.net)

calculate_submission(net, X)
