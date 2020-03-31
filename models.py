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


def CNN(no_inputs, no_outputs, pooling_type):
    
    """
        (#filters, kernel size, stride, activation)
    
    """
    NEIGHBORS_A = 16
    NEIGHBORS_D = 19
    
    A_mat = np.load('conv_mat_A.npy')
    D_mat = np.load('conv_mat_D.npy')
    
    inputs = Input(shape=(no_inputs,), dtype='int32')
    
    A_in = tf.matmul(inputs, A_mat)
    D_in = tf.matmul(inputs, D_mat)
    
    x_A = Conv1D(162, NEIGHBORS_A, NEIGHBORS_A, activation='relu')(A_in)
    x_D = Conv1D(162, NEIGHBORS_D, NEIGHBORS_D, activation='relu')(D_in)
    
    x_A = Conv1D(16, 4, 2, activation='relu')(x_A)
    x_D = Conv1D(16, 4, 2, activation='relu')(x_D)
    
    #behöver mer arbete, trasig
    
    outputs = 0
    
    return Model(inputs, outputs)

