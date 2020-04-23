""" @PETER HALLDESTAM, 2020

    models to analyze
    
"""

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.layers import Concatenate
import tensorflow.keras.backend as K

import tensorflow as tf
import numpy as np


from layers import GraphConv
from transformations import get_adjacency_matrix

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



    



def CNN(no_inputs, no_outputs, depth=3, width=80, filters = [256, 16, 4],
        sort = 'CCT', rotations = True, reflections = True):
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
    
    NEIGHBORS_A, NEIGHBORS_D = 16, 19
    
    
    if rotations:
        no_rotations_A, no_rotations_D = 5, 6
        rot = "_rot"
    else:
        no_rotations_A, no_rotations_D = 1, 1
        rot = ""
    
    if reflections:
        refl_mult = 2
        refl = "_refl"
    else:
        refl_mult = 1
        refl = ""
    
    MAT_PATH = 'ConvolutionalMatrix/'
    
    A_mat = np.load(MAT_PATH+'A_mat_'+sort+rot+refl+'.npy')
    D_mat = np.load(MAT_PATH+'D_mat_'+sort+rot+refl+'.npy')
    
    inputs = Input(shape=(no_inputs,), dtype='float32')
    
    A_in = tf.matmul(inputs, A_mat)
    D_in = tf.matmul(inputs, D_mat)
    
    #reshapes the input to [batch, steps, channels]
    
    A_in = tf.reshape(A_in, [-1, A_in.shape[1], 1])
    D_in = tf.reshape(D_in, [-1, D_in.shape[1], 1])
   
    #parameters for conv1D: filters, kernel size, stride, activation
    
    x_A = Conv1D(filters[0], NEIGHBORS_A, NEIGHBORS_A, activation='relu', 
                 input_shape = (None, A_in.shape[1], 1), data_format = "channels_last" )(A_in)
    
    x_D = Conv1D(filters[0], NEIGHBORS_D, NEIGHBORS_D, activation='relu', 
                 input_shape = (None, D_in.shape[1], 1), data_format = "channels_last" )(D_in)
    
    x_A = Conv1D(filters[1], no_rotations_A, no_rotations_A, activation='relu')(x_A)
    x_D = Conv1D(filters[1], no_rotations_D, no_rotations_D, activation='relu')(x_D)
    
    #x_A = MaxPooling1D(pool_size=2)(x_A)
    #x_D = MaxPooling1D(pool_size=2)(x_D)
    
    x_A = Flatten()(x_A)
    x_D = Flatten()(x_D)
    
    FCN_in = Concatenate(axis=1)([x_A, x_D])
    
    x = Dense(width, activation='relu')(FCN_in)
    
    for i in range(depth-1):
        x = Dense(width, activation='relu')(x)
        
    outputs = Dense(no_outputs, activation='linear')(x)
    
    return Model(inputs, outputs)

def CN_FCN(no_inputs, no_outputs, depth=[3, 3], width=80, filters = [32, 16],
        sort = 'CCT', rotations = True, reflections = True):
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
    
    NEIGHBORS_A, NEIGHBORS_D = 16, 19
    
    
    if rotations:
        no_rotations_A, no_rotations_D = 5, 6
        rot = "_rot"
    else:
        no_rotations_A, no_rotations_D = 1, 1
        rot = ""
    
    if reflections:
        refl_mult = 2
        refl = "_refl"
    else:
        refl_mult = 1
        refl = ""
    
    MAT_PATH = 'ConvolutionalMatrix/'
    
    A_mat = np.load(MAT_PATH+'A_mat_'+sort+rot+refl+'.npy')
    D_mat = np.load(MAT_PATH+'D_mat_'+sort+rot+refl+'.npy')
    
    inputs = Input(shape=(no_inputs,), dtype='float32')
    
    A_in = tf.matmul(inputs, A_mat)
    D_in = tf.matmul(inputs, D_mat)
    
    #reshapes the input to [batch, steps, channels]
    
    A_in = tf.reshape(A_in, [-1, A_in.shape[1], 1])
    D_in = tf.reshape(D_in, [-1, D_in.shape[1], 1])
   
    #parameters for conv1D: filters, kernel size, stride, activation
    
    x_A = Conv1D(filters[0], NEIGHBORS_A, NEIGHBORS_A, activation='relu', name='Conv_cluster_A',
                 input_shape = (None, A_in.shape[1], 1), data_format = "channels_last" )(A_in)
    
    x_D = Conv1D(filters[0], NEIGHBORS_D, NEIGHBORS_D, activation='relu', name='Conv_cluster_D',
                 input_shape = (None, D_in.shape[1], 1), data_format = "channels_last" )(D_in)
    
    #FCN-tjosan
    for i in range(depth[0]):
        x_A = Conv1D(filters[0], 1, 1, activation = 'relu', name = 'FCN_emu_A' + str(i+1))(x_A)
        x_D = Conv1D(filters[0], 1, 1, activation = 'relu', name = 'FCN_emu_D' + str(i+1))(x_D)
    
    x_A = Conv1D(filters[1], no_rotations_A, no_rotations_A*refl_mult, 
                 activation='relu', name = 'Conv_orientation_A')(x_A)
    x_D = Conv1D(filters[1], no_rotations_D, no_rotations_D*refl_mult,
                 activation='relu', name = 'Conv_orientation_D')(x_D)
    
    #x_A = MaxPooling1D(pool_size=2)(x_A)
    #x_D = MaxPooling1D(pool_size=2)(x_D)
    
    x_A = Flatten()(x_A)
    x_D = Flatten()(x_D)
    
    FCN_in = Concatenate(axis=1)([x_A, x_D])
    
    x = Dense(width, activation='relu')(FCN_in)
    
    for i in range(depth[1]-1):
        x = Dense(width, activation='relu')(x)
        
    outputs = Dense(no_outputs, activation='linear')(x)
    
    return Model(inputs, outputs)
