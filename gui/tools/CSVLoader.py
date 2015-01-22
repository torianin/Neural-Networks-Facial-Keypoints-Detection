import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle


def load(ftrain='/Users/torianin/Neural-Networks-Facial-Keypoints-Detection/training.csv', ftest='/Users/torianin/Neural-Networks-Facial-Keypoints-Detection/test.csv', test=False, cols=None):
    fname = ftest if test else ftrain
    df = read_csv(os.path.expanduser(fname)) 

    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:
        df = df[list(cols) + ['Image']]

    print(df.count())
    df = df.dropna()

    X = np.vstack(df['Image'].values) / 255.
    X = X.astype(np.float32)

    if not test:
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48 
        X, y = shuffle(X, y, random_state=42)
        y = y.astype(np.float32)
    else:
        y = None

    return X, y