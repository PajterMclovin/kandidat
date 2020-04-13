""" @PETER HALLDESTAM, 2020

    models to analyze
    
"""

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.layers import Concatenate
import tensorflow.keras.backend as K

import tensorflow as tf
import numpy as np


from layers import GraphConv
from transformations import get_adjacency_matrix, reduce_columns

def FCN(no_inputs, no_outputs, no_layers, no_nodes,
        cartesian=False, classification=False):
    """
    Args:
        no_inputs  : number of input nodes
        no_outputs : number of ouput nodes
        no_layers  : number of fully-connected layers
        no_nodes   : number of nodes in each layer
        cartesian_coordinates : matters only with classification nodes
        classifcation_nodes : True if training with classification nodes
    Returns:
        fully-connected neural network as tensorflow.keras.Model object
    """
    inputs = Input(shape=(no_inputs,), dtype='float32')
    x = Dense(no_nodes, activation='relu')(inputs)
    for i in range(no_layers-2):
        x = Dense(no_nodes, activation='relu')(x)
        
    if classification:
        no_classifications = int(no_outputs/4)
        no_regression = no_outputs-no_classifications
    
        output1 = Dense(no_regression, activation='linear')(x)                 #for regression
        output2 = Dense(no_classifications, activation='sigmoid')(x)           #for classification
        outputs = Concatenate(axis=1)([output1, output2])
        
    else:
        outputs = Dense(no_outputs, activation='linear')(x)
        
    return Model(inputs, outputs)



def GCN(no_inputs, no_outputs, no_layers, no_nodes):
    
    inputs = Input(shape=(no_inputs,), dtype='float32')
    
    x = GraphConv(no_inputs, activation='relu')(inputs)
    x = GraphConv(no_inputs, activation='relu')(x)
    
    x = Dense(no_nodes, activation='relu')(inputs)
    for i in range(no_layers-2):
        x = Dense(no_nodes, activation='relu')(x)
    
    outputs = Dense(no_outputs, activation='linear')(x)
    return Model(inputs, outputs)



    



def CNN(no_inputs, no_outputs, depth=2, width=8, pooling_type=0):
    """

    Parameters
    ----------
    no_inputs : int
        number of input nodes.
    no_outputs : int
        number of ouput nodes.
    depth : int, optional
        depth of FCN-bit. The default is 2.
    width : int, optional
        width of FCN-bit. The default is 8.
    pooling_type : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    NEIGHBORS_A = 16
    NEIGHBORS_D = 19
    
    print("Shaping inputs...")
    
    A_mat = reduce_columns(np.load('conv_mat_A.npy'))
    D_mat = reduce_columns(np.load('conv_mat_D.npy'))
    
    inputs = Input(shape=(no_inputs,), dtype='float32')
    
    A_in = tf.matmul(inputs, A_mat)
    D_in = tf.matmul(inputs, D_mat)
    
    #reshapes the input to [batch, steps, channels]
    
    A_in = tf.reshape(A_in, [-1, A_in.shape[1], 1])
    D_in = tf.reshape(D_in, [-1, D_in.shape[1], 1])
   
    #parameters for conv1D: filters, kernel size, stride, activation

    x_A = Conv1D(8, NEIGHBORS_A, NEIGHBORS_A, activation='relu', 
                 input_shape = (None, no_inputs*NEIGHBORS_A, 1), data_format = "channels_last" )(A_in)
    
    x_D = Conv1D(8, NEIGHBORS_D, NEIGHBORS_D, activation='relu', 
                 input_shape = (None, no_inputs*NEIGHBORS_D, 1), data_format = "channels_last" )(D_in)
    
    x_A = Conv1D(4, 9, 3, activation='relu')(x_A)
    x_D = Conv1D(4, 9, 3, activation='relu')(x_D)
    
    x_A = Flatten()(x_A)
    x_D = Flatten()(x_D)
    
    FCN_in = Concatenate(axis=1)([x_A, x_D])
    
    x = Dense(width, activation='relu')(FCN_in)
    
    for i in range(depth-1):
        x = Dense(width, activation='relu')(x)
        
    outputs = Dense(no_outputs, activation='linear')(x)
    
    return Model(inputs, outputs)

