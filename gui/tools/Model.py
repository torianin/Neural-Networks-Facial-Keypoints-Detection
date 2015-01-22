from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

def get_neural_net():
    return NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],

    input_shape=(None, 9216),
    hidden_num_units=100,  
    output_nonlinearity=None,
    output_num_units=30,  

    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,  
    max_epochs=400,  
    verbose=1,
    )