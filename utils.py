""" @PETER HALLDESTAM, 2020
    Help methods used in neural_network.py
    
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.backend import variable
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from itertools import permutations
from math import factorial

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
    

def plot_predictions(prediction, labels, bins=500):
    """
    Use to plot a models predictions in similar format as previous years, i.e. 2d histograms ("lasersvärd")
    
    Args:
        prediction : use model.predict(data)
        labels : to compare with predictions
    Returns:
        figure, axes, events (dict)
    Raises : if data and labels is not the same length
        ValueError : if prediction and labels is not of same length
    """
    if not len(prediction)==len(labels):
        raise TypeError('The prediction must be of same length as labels.') 
        
    events = {'predicted_energy': prediction[::,0::3].flatten(),
              'correct_energy': labels[::,0::3].flatten(), 
              
              'predicted_theta': np.mod(prediction[::,1::3],np.pi).flatten(),
              'correct_theta': labels[::,1::3].flatten(),
              
              'predicted_phi': np.mod(prediction[::,2::3],2*np.pi).flatten(),
              'correct_phi': labels[::,2::3].flatten()}
    
    figure, axes = plt.subplots(1,3, figsize=(13, 5))
    image = []
    image.append(axes[0].hist2d(events['correct_energy'], events['predicted_energy'], bins=bins, norm=LogNorm(), cmax = 1001))
    image.append(axes[1].hist2d(events['correct_theta'], events['predicted_theta'], bins=bins, norm=LogNorm(), cmax = 1001))
    image.append(axes[2].hist2d(events['correct_phi'], events['predicted_phi'], bins=bins, norm=LogNorm(), cmax = 1001))
    figure.tight_layout()
    return figure, axes, events

def plot_predictions_cart(prediction, labels, bins=500):
    """
    Use to plot a models predictions, with cartesian coordinates
    
    Args:
        prediction : use model.predict(data)
        labels : to compare with predictions
    Returns:
        figure, axes, events (dict)
    Raises : if data and labels is not the same length
        ValueError : if prediction and labels is not of same length
    """
    if not len(prediction)==len(labels):
        raise TypeError('The prediction must be of same length as labels.') 
        
    events = {'predicted_energy': prediction[::,0::4].flatten(),
              'correct_energy': labels[::,0::4].flatten(), 
              
              'predicted_x': prediction[::,1::4].flatten(),
              'correct_x': labels[::,1::4].flatten(),
              
              'predicted_y': prediction[::,2::4].flatten(),
              'correct_y': labels[::,2::4].flatten(),
              
              'predicted_z': prediction[::,3::4].flatten(),
              'correct_z': labels[::,3::4].flatten()}
    
    figure, axes = plt.subplots(1,4, figsize=(13, 5))
    image = []
    image.append(axes[0].hist2d(events['correct_energy'], events['predicted_energy'], bins=bins, norm=LogNorm(), cmax = 1001))
    image.append(axes[1].hist2d(events['correct_x'], events['predicted_x'], bins=bins, norm=LogNorm(), cmax = 1001))
    image.append(axes[2].hist2d(events['correct_y'], events['predicted_y'], bins=bins, norm=LogNorm(), cmax = 1001))
    image.append(axes[3].hist2d(events['correct_z'], events['predicted_z'], bins=bins, norm=LogNorm(), cmax = 1001))
    figure.tight_layout()
    return figure, axes, events
    


def plot_accuracy(history):
    """
    Learning curve; plot training and validation ACCURACY from model history
    
    Args:
        history : tf.keras.callbacks.History object returned from model.fit
    Returns:
        -
    Raises:
        TypeError : if history is not History object
    """
    if not isinstance(history, tf.keras.callbacks.History):
        raise TypeError('history must of type tf.keras.callbacks.History')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

def plot_loss(history):
    """
    Learning curve; plot training and validation LOSS from model history
    
    Args:
        history : tf.keras.callbacks.History object returned from model.fit
    Returns:
        -
    Raises:
        TypeError : if history is not History object
    """
    if not isinstance(history, tf.keras.callbacks.History):
        raise TypeError('history must of type tf.keras.callbacks.History')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')


def get_permutation_tensor(n):
    P = np.zeros((factorial(n), 3*n, 3*n))
    depth = 0
    for perm in permutations(range(n)):    
        for i in range(n):
            P[depth, 3*i:3*i+3:, 3*perm[i]:3*perm[i]+3:] = np.identity(3)
        depth += 1
    return variable(P)

def get_identity_tensor(n):
    I = np.zeros((factorial(n), 3*n, 3*n))
    for depth in range(factorial(n)):
            for i in range(n):
                I[depth, 3*i:3*i+3, 3*i:3*i+3] = np.identity(3)
    return variable(I)

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
