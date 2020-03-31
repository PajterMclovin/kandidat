""" @PETER HALLDESTAM, 2020

    Help methods used in neural_network.py
    

"""
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from itertools import permutations
from random import shuffle

from loss_functions import Loss_function
from loss_functions import get_identity_tensor
from loss_functions import get_permutation_tensor
from tensorflow import transpose

def load_data(npz_file_name, total_portion, cartesian_coordinates=False):
    """
    Reads a .npz-file containing simulation data in spherical coordinates and
    returns data/labels in spherical or cartesian coordinates in numpy arrays
    
    Args:
        npz_file_name : address string to .npz-file
        total_portion : the actual amount of total data to be used
        cartesian_coordinates : set True for cartesian coordinates
    Returns:
        all data and labels in spherical (or cartesian) coordinates
    Raises:
        ValueError : if portion is not properly set
    """
    if not total_portion > 0 and total_portion <= 1:
        raise ValueError('total_portion must be in the interval (0,1].')
    print('Reading data...')
    data_set = np.load(npz_file_name)
    det_data = data_set['detector_data']
    labels = data_set['energy_labels']
    if cartesian_coordinates:
        print('Transforming to cartesian coordinates')
        labels = spherical_to_cartesian(labels)
    no_events = int(len(labels)*total_portion)
    print('Using {} simulated events in total.'.format(no_events))
    return det_data[:no_events], labels[:no_events]


def get_eval_data(data, labels, eval_portion=0.1):
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
    Sort the label data matrix with rows (energy1, theta1, phi1, ... phiM), 
    where M is the max multiplicity.
    
    Args:
        old_npz : address of .npz-file with label data to sort
        new_npz : name of new .npz file
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


def add_empty_events(old_npz, new_npz, n):
    """
    Appends n empty events into data/label, shuffles them and then saves it to
    a new npz-file. Surely not the optimal solution but it should work. Maybe
    fucks up with larger data sets...
    
    Args:
        old_npz : address of .npz-file with label data to sort
        new_npz : name of new .npz file
    """
    data_set = np.load(old_npz)
    data = data_set['detector_data']
    labels = data_set['energy_labels']
    
    no_inputs = len(data[0])
    no_outputs = len(labels[0])
    data_list = np.vstack([data, np.zeros((n, no_inputs))]).tolist()
    label_list = np.vstack([labels, np.zeros((n, no_outputs))]).tolist()
    
    tmp = list(zip(data_list, label_list))
    shuffle(tmp)
    new_data, new_labels = zip(*tmp)
    np.savez(new_npz, detector_data=new_data, energy_labels=new_labels)
    


def get_permutation_match(y, y_, cartesian_coordinates=False, loss_type='mse'):
    """
    Sorts the predictions with corresponding label as the minimum of a square 
    error loss function. Must be used BEFORE plotting the "lasersvärd".

    """
    m = 3
    if cartesian_coordinates: 
        m = 4
    max_mult = int(len(y_[0])/m)
    permutation_tensor = get_permutation_tensor(max_mult, m=m)
    identity_tensor = get_identity_tensor(max_mult, m=m)
    
    #get all possible combinations
    Y_ = transpose(K.dot(K.variable(y_), permutation_tensor), perm=[1,0,2])
    Y = transpose(K.dot(K.variable(y), identity_tensor), perm=[1,0,2])
    
    loss_function = Loss_function(loss_type, cartesian_coordinates=cartesian_coordinates)
    permutation_indices = K.argmin(K.sum(loss_function.loss(Y, Y_), axis=2), axis=0)
    
    print('Matching predicted data with correct label permutation. May take a while...')
    for i in range(len(y_)):
        y_[i,::] = Y_[permutation_indices[i],i,::]    
    return y, y_



def spherical_to_cartesian(spherical_labels):
    """
    Coordinate transform (theta, phi) --> (x,y,z). Used for labels before training
    
    """
    max_mult = int(len(spherical_labels[0])/3)
    cartesian_labels = np.zeros([len(spherical_labels), 4*max_mult])
    energy = spherical_labels[::,0::3]
    
    theta = spherical_labels[::,1::3]
    phi = spherical_labels[::,2::3]

    x = np.sin(theta)*np.cos(phi)    
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)

    cartesian_labels[::, 0::4] = energy
    cartesian_labels[::, 1::4] = x
    cartesian_labels[::, 2::4] = y
    cartesian_labels[::, 3::4] = z
    return cartesian_labels
    

def cartesian_to_spherical(cartesian_labels):
    """
    Coordinate transform (x,y,z) --> (theta, phi). Used for labels and predictions
    after training

    """
    max_mult = int(len(cartesian_labels[0])/4)
    spherical_labels = np.zeros([len(cartesian_labels), 3*max_mult])
    energy = cartesian_labels[::,0::4]
    
    x = cartesian_labels[::,1::4]
    y = cartesian_labels[::,2::4]
    z = cartesian_labels[::,3::4]
    r = np.sqrt(x*x+y*y+z*z)
    print(np.min(r))
    
    theta = np.arccos(z/r)
    phi = np.arctan2(y,x)
    
    spherical_labels[::,0::3] = energy
    spherical_labels[::,1::3] = theta
    spherical_labels[::,2::3] = np.mod(phi, 2*np.pi)
    return spherical_labels
    
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
