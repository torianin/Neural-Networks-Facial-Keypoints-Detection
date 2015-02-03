import cPickle as pickle

def store_model(net, filename):
    with open(filename, 'wb') as f:
        pickle.dump(net, f, -1)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
