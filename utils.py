import cPickle as pickle
import numpy as np
import pandas
from specialist_settings import *

def store_model(net, filename):
    with open(filename, 'wb') as f:
        pickle.dump(net, f, -1)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def feature_idx(feature_name):
    features = "left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y,left_eye_inner_corner_x,left_eye_inner_corner_y,left_eye_outer_corner_x,left_eye_outer_corner_y,right_eye_inner_corner_x,right_eye_inner_corner_y,right_eye_outer_corner_x,right_eye_outer_corner_y,left_eyebrow_inner_end_x,left_eyebrow_inner_end_y,left_eyebrow_outer_end_x,left_eyebrow_outer_end_y,right_eyebrow_inner_end_x,right_eyebrow_inner_end_y,right_eyebrow_outer_end_x,right_eyebrow_outer_end_y,nose_tip_x,nose_tip_y,mouth_left_corner_x,mouth_left_corner_y,mouth_right_corner_x,mouth_right_corner_y,mouth_center_top_lip_x,mouth_center_top_lip_y,mouth_center_bottom_lip_x,mouth_center_bottom_lip_y".split(",")
    return features.index(feature_name)

def find_specialist(feature_name):
    for idx, s in enumerate(SPECIALIST_SETTINGS):
        if feature_name in s['columns']:
            return idx
    return None

def predict_feature(main_net, specialists, feature_name, X):
    specialist_idx = find_specialist(feature_name)
    print specialist_idx
    if specialist_idx:
        model = specialists[specialist_idx]
        fidx = SPECIALIST_SETTINGS[specialist_idx]['columns'].index(feature_name)
    else:
        model = main_net
        fidx = feature_idx(feature_name)
    return model.predict(X)[:, fidx]*48+48

def predict(main_net, specialists, X):
    return np.vstack([predict_feature(main_net, specialists, f, X) for f in FEATURE_NAMES]).T

def load_data(flip=True):
    data = pandas.io.parsers.read_csv("training.csv")
    data['Image'] = data['Image'].apply(lambda x: np.array(x.split(), dtype=np.float32))
    X = np.vstack(data['Image'].values) / 255.
    y = data[data.columns[:-1]].values
    y = (y - 48) / 48
    y = y.astype(np.float32)

    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]
    if flip:
        X2 = np.fliplr(X)
        y2 = np.copy(y)
        for a,b in flip_indices:
            y2[:, [a, b]] = y2[:, [b,a]]
        X = np.vstack([X,X2])
        y = np.vstack([y,y2])

    p = np.random.permutation(len(X))
    X = X[p]
    y = y[p]
    X = X.reshape(-1, 1, 96, 96)
    return X, y

def load_specialists():
    return [load_model("specialist_"+s['name']+".pickle") for s in SPECIALIST_SETTINGS]

def filter_data(X, y, columns):
    y = y[:, columns]
    good_rows = np.all(np.isfinite(y), axis=1)
    return X[good_rows], y[good_rows]

