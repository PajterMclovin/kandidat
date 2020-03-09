#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:50:25 2019

@author: rickardkarlsson
"""

"""
    Conventions I use:
        x is always the input that is sent into the ANN (artificial neural network)
"""
# Importing installed modules (At the end of the file I import other modules from github repo)
import numpy as np
import tensorflow as tf
import sys
import importlib, importlib.util
import scipy
import math

#### Variables #####
input_size = 162
neighbours_A = 16
neighbours_D = 19

nbr_A_crystals = 12
nbr_D_crystals = 30

#############################################################
############ Convolution layer methods ######################
#############################################################
#


def hidden_conv_layers_without_conv_mat(x_input, stride, filter_size, no_filter_per_layer,
                                        crystal_type = None, first_layer_conv_mat = True,
                                        c_matrix = None, nbr_of_rotations = 1):

    """
        Used for a certain update of the convolution matrices that are much smaller in size

        Constructs convuluted layers
        x = input data
        stride = the stride for convoluting, except for the conv_matrix layer
        no_nodes_per_layer = input shape either scalar number or array (eg. [182, 34, 35]) with length {no_hidden_layers}
        no_output_nodes  = number of output nodes, for example 2*{maximum multiplicity of the batch}

        crystal_type decides what crystal that is the input, 'A' or 'D'
        first_conv_layer_mat is True if this is the first time this method is used for a crystal,
        otherwise False
        c_matrix = convolution matrix, only used in first layer if first_conv_layer_mat is True

    """

    no_hidden_layers = len(no_filter_per_layer)

    ## Just a cheaty quick fix
    if crystal_type == 'B' or crystal_type == 'C':
        crystal_type = 'D'

    if crystal_type != 'A' and crystal_type != 'D' and crystal_type != None:
        raise ValueError('Invalid crystal type')


    ## Initial layer
    if first_layer_conv_mat == True:
        if crystal_type == 'A':
            first_filter_size = neighbours_A
        if crystal_type == 'D':
            first_filter_size = neighbours_D
        if crystal_type == None:
            raise ValueError('Invalid crystal type, can not be None')

        x_output = conv_layer(x_input, 1, no_filter_per_layer[0],
                                first_filter_size, c_matrix,
                                crystal_type=crystal_type)
        image_shape = [-1, 1, 162*nbr_of_rotations, no_filter_per_layer[0]]
        x_output = tf.reshape(x_output, image_shape)
    else:
        prev_no_filters = int(x_input.shape[3])
        x_output = conv_layer_without_conv_mat(x_input,prev_no_filters,no_filter_per_layer[0],
                                               filter_size, stride)

    # Middle and last layers
    for i in range(1,no_hidden_layers):
        x_output = conv_layer_without_conv_mat(x_output,no_filter_per_layer[i-1],
                                                no_filter_per_layer[i],
                                                filter_size, stride)
    return x_output

def conv_layer(x_input, prev_nbr_of_filter, nbr_of_filter, filter_size, c_matrix, crystal_type = None,
               last_layer = False, nbr_rotations = 1, sparse = False):
    """
    INPUT
    x_input (array [-1,X]) The input to the neural layer
    prev_nbr_of_filter (int)  The number of filters in the previous layer
    nbr_of_filter (int) The number of filters for this layer
    filter_size (int) The size of the filter, will also decide the stride for the convolution
    c_matrix (np matrix) The matrix containing information about neighbouring crystals,
                         used to create image
    crystal_type is either 'A' or 'D' but it is not needed currently
    last_layer (boolean) False if not last layer in your CNN or True if it is the last, affects output size

    OUTPUT
    The output of the network layer as a row vector
    """
    weights, bias = construct_filter(x_input, prev_nbr_of_filter, nbr_of_filter, filter_size)
    conv_image = construct_conv_image(x_input, c_matrix, filter_size, prev_nbr_of_filter, sparse)
    layer_output = tf.nn.relu(conv2d(conv_image,weights,filter_size)+bias)
    if crystal_type == 'A':
        x_size = nbr_A_crystals
    elif crystal_type == 'D':
        x_size = nbr_D_crystals
    else:
        x_size = input_size

    if last_layer is True:
        layer_output= tf.reshape(layer_output, [-1, x_size*nbr_of_filter*nbr_rotations])
    else:
        layer_output= tf.reshape(layer_output, [-1, x_size])
    return layer_output

def conv_layer_without_conv_mat(x_input, prev_nbr_of_filter, nbr_of_filter,
                                filter_size,  stride,
                                last_layer = False):
    """
    Used for a certain update of the convolution matrices that are much smaller in size

    INPUT
    x_input (array [-1,X]) The input to the neural layer
    prev_nbr_of_filter (int)  The number of filters in the previous layer
    nbr_of_filter (int) The number of filters for this layer
    filter_size (int) The size of the filter, will also decide the stride for the convolution
    last_layer (boolean) False if not last layer in your CNN or True if it is the last, affects output size DONT USE


    OUTPUT
    The output if the network layer as a row vector
    """

    weights, bias = construct_filter(x_input, prev_nbr_of_filter, nbr_of_filter, filter_size)
    layer_output = tf.nn.relu(conv2d(x_input,weights,stride)+bias)

    return layer_output


def construct_filter(x_input, prev_nbr_of_filter, nbr_of_filters, filter_size):
    """
    INPUT
    See conv_layer for information about inputs

    OUTPUT
    The weights and bias for the filters
    """
    filter_shape = [1, filter_size, prev_nbr_of_filter, nbr_of_filters]
    weights = tf.Variable(tf.truncated_normal(shape = filter_shape, stddev=0.1, dtype=tf.float32))
    bias = tf.Variable(tf.constant(0.1, shape=[nbr_of_filters]))
    return weights, bias


def construct_conv_image(x_input, c_matrix, filter_size, prev_nbr_of_filter, sparse):
    """
    INPUT
    See conv_layer for information about inputs

    OUTPUT
    The image that contains information about the geometry and neighbours
    """
    if sparse:
        conv_image = tf.sparse_tensor_dense_matmul(c_matrix,x_input, adjoint_a = True, adjoint_b = True)
        conv_image = tf.transpose(conv_image)
    else:
        conv_image = tf.matmul(x_input,c_matrix)
    image_shape = [-1, 1, int(conv_image.shape[1]), prev_nbr_of_filter]
    conv_image = tf.reshape(conv_image, image_shape)
    return conv_image


def conv2d(x, weights, stride):
    """
        The actual convolutional operation

    INPUT
    x: The image to be convoluted
    weights: The filters weight
    stride: size of stride for the filter

    OUTPUT
    The convoluted image
    """
    return tf.nn.conv2d(x, weights, strides=[1,1,stride,1], padding='VALID')

def dense_2_sparse(matrix):
    """
        Creates and outputs a sparse tensor from a ordinary dense numpy matrix
    """
    matrix = scipy.sparse.coo_matrix(matrix)
    row = matrix.row
    col = matrix.col
    ind = np.empty([len(row),2])
    count = 0
    for i in row:
        ind[count,0] = i
        count += 1
    count = 0
    for j in col:
        ind[count,1] = j
        count += 1

    ind = tf.convert_to_tensor(ind, dtype=tf.int64)
    val = tf.convert_to_tensor(matrix.data, dtype=tf.float32)
    shape = tf.convert_to_tensor(matrix.shape, dtype=tf.int64)

    sparse_tensor = tf.SparseTensor(indices = ind, values = val, dense_shape= shape)
    return sparse_tensor

#############################################################
############   Data preparation methods           ###########
#############################################################

def gen_sub_set(batch_size, batch_x, batch_y):
    """
        Returns a randomly selected subset of batch_x and batch_y with length batch_size.
    """
    if not len(batch_x) == len(batch_y):
        raise ValueError('Lists most be of same length')
    indices = np.random.randint(0, len(batch_x), size=batch_size)
    return batch_x[indices], batch_y[indices]

#############################################################
############   Module importing method           ############
#############################################################

def module_from_file(module_name, file_path):
    """
        Used to import modules from different folder in our Github repo.

        module_name is the file name (Ex. 'file.py')
        file_path is the file path (Ex. '../save_load/file.py) if you want to reach a
        folder in parent directory

        Outputs the module as variable that you use as an usual import reference
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
