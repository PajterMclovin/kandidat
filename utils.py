""" @PETER HALLDESTAM, 2020

    Help methods used in neural_network.py
    

"""
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from itertools import permutations
from math import factorial

from loss_functions import LAMBDA_ENERGY, LAMBDA_THETA, LAMBDA_PHI


def load_data(npz_file_name, total_portion):
    """
    Reads the relevant data from a npz file for an energy/theta output network.
    
    Args:
        npz_file_name : address string to .npz-file containing simulation data
        total_portion : the actual amount of total data to be used
    Returns:
        all data and labels
    Raises:
        ValueError : if portion is not properly set
    """
    if not total_portion > 0 and total_portion <= 1:
        raise ValueError('total_portion must be in the interval (0,1].')
    print('Reading data...')
    data_set = np.load(npz_file_name)
    det_data = data_set['detector_data']
    labels = data_set['energy_labels']
    no_events = int(len(labels)*total_portion)
    print('Using {} simulated events in total.'.format(no_events))
    return det_data[:no_events], labels[:no_events]


def get_eval_data(data, labels, eval_portion):
    """
    Detach final evaluation data.
    
    Args:
        data : dataset to split in two
        labels : corresponding data labels
        eval_portion : the amount of data for final evaluation
    Returns:
        training/validation data and labels
        final evaluation data and labels
    Raises:
        ValueError : if portion is not properly set
    """
    if not eval_portion > 0 and eval_portion < 1:
        raise ValueError('eval_portion must be in interval (0,1).')
    no_eval = int(len(data)*eval_portion)
    return data[no_eval:], labels[no_eval:], data[:no_eval], labels[:no_eval]


def sort_data(old_npz, new_npz):
    """
    Sort the label data matrix with rows (energy1, theta1, phi1, ... phiM), where M is the max multiplicity.
    
    Args:
        (string) old_npz : address of .npz-file with label data to sort
        (string) new_npz : name of new .npz file
    Returns:
        numpy array with the sorted data
    """
    data_set = np.load(old_npz)
    labels = data_set['energy_labels']
    
    max_mult = int(len(labels[0])/3)
    energies = labels[::,::3]
    sort_indices = np.argsort(-energies) 
    sorted_labels = np.zeros([len(labels), 3*max_mult])
    for i in range(len(labels)):
        for j in range(max_mult):
            sorted_labels[i,3*j:3*(j+1)] = labels[i,3*sort_indices[i,j]:3*(sort_indices[i,j]+1)]
    np.savez(new_npz, detector_data=data_set['detector_data'], energy_labels=sorted_labels)
    return sorted_labels


def get_permutation_tensor(n):
    """
    Returns a tensor containing all n! permutation matrices. Dimension: [n!, 3n, 3n]
    """
    permutation_tensor = np.zeros((factorial(n), 3*n, 3*n))
    depth = 0
    for perm in permutations(range(n)):    
        for i in range(n):
            permutation_tensor[depth, 3*i:3*i+3:, 3*perm[i]:3*perm[i]+3:] = np.identity(3)
        depth += 1
    return K.variable(permutation_tensor)


def get_identity_tensor(n):
    """
    Returns a tensor containing n! identity matrices. Dimension: [n!, 3n, 3n]
    """
    identity_tensor = np.zeros((factorial(n), 3*n, 3*n))
    for depth in range(factorial(n)):
            for i in range(n):
                identity_tensor[depth, 3*i:3*i+3, 3*i:3*i+3] = np.identity(3)
    return K.variable(identity_tensor)


def get_permutation_match(y, y_):
    """
    Sorts the predictions with corresponding label as the minimum of a square 
    error loss function. Must be used BEFORE plotting the "lasersvärd".
    
    Args:
        y : use model.predict(data)
        y_ : to compare with predictions y and sort
    Returns:
        y : same as input y
        y_ : sorted labels
    """
    max_mult = int(len(y_[0])/3)
    
    def get_matching_label(x, x_):
        min_loss = np.inf
        for p in permutations(range(max_mult), max_mult):
            perm_loss = 0
            for i in range(max_mult):
                energy_loss = LAMBDA_ENERGY*np.power(x[3*i] - x_[3*p[i]], 2)
                theta_loss = LAMBDA_THETA*np.power(np.mod(x[3*i+1], 2*np.pi) - x_[3*p[i]+1], 2)
                phi_loss = LAMBDA_PHI*np.power(np.mod(x[3*i+2], 2*np.pi) - x_[3*p[i]+2], 2)
                perm_loss += energy_loss + theta_loss + phi_loss
            if perm_loss < min_loss:
                min_loss = perm_loss
                perm = p
        permutation_matrix = np.zeros((3*max_mult, 3*max_mult))
        for i in range(max_mult):
            permutation_matrix[3*i:3*i+3:, 3*perm[i]:3*perm[i]+3:] = np.identity(3)
        return np.dot(x_, permutation_matrix)
    
    print('Matching predicted data with correct label permutation. May take a while...')
    for i in range(len(y_)):
        y_[i] = get_matching_label(y[i], y_[i])
    print('done!')
    return y, y_

### ---------------- Davids code ----------------------------------------------

def data_to_cartesian(old_npz, new_npz):
    """
    Converts the energy_labels of the of the .npz-file to cartesian coordinates
    
    Parameters
    ----------
    old_npz : string
        file whose energy labels to convert to cartesian coordinates
    new_npz : string
        file name for new npz-file
    Returns
    -------
    None.
    """
    data_set = np.load(old_npz)
    labels = data_set['energy_labels']

    max_mult = int(len(labels[0])/3)

    theta = labels[::, 1::3]
    phi = labels[::, 2::3]

    x = tf.math.multiply(tf.math.sin(theta), tf.math.cos(phi))
    y = tf.math.multiply(tf.math.sin(theta), tf.math.sin(phi))
    z = tf.math.cos(theta)

    cart_labels = np.zeros([len(labels), 4*max_mult])

    cart_labels[::, 0::4] = labels[::, 0::3]
    cart_labels[::, 1::4] = x
    cart_labels[::, 2::4] = y
    cart_labels[::, 3::4] = z

    np.savez(new_npz, detector_data=data_set['detector_data'], energy_labels=cart_labels)
    return cart_labels

    
    
### ----------------------------- INSPIRATION ---------------------------------
    
#Need custom conv-Layer or some other solution before we can implement ResNet. BatchNorm is easy to implement!
def res_net_block(input_data, filters, conv_size):
    x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, input_data])
    x = layers.Activation('relu')(x)
    return x

#For comparison with ResNet. Is there any difference? If so we'll might be able to build deep ass neural networks!!
def non_res_block(input_data, filters, conv_size):
    x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x
