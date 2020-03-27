""" @PETER HALLDESTAM, 2020

    models to analyze
    
"""

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense


def FCN(no_inputs, no_outputs, no_layers, no_nodes):
    """
    Args:
        no_inputs  : number of input nodes
        no_outputs : number of ouput nodes
        no_layers  : number of fully-connected layers
        no_nodes   : number of nodes in each layer
    Returns:
        fully-connected neural network as tensorflow.keras.Model object
    """
    inputs = Input(shape=(no_inputs,), dtype='float32')
    x = Dense(no_nodes, activation='relu')(inputs)
    for i in range(no_layers):
        x = Dense(no_nodes, activation='relu')(x)
    outputs = Dense(no_outputs, activation='relu')(x)
    return Model(inputs, outputs)
    

