""" @PETER HALLDESTAM, 2020

    Help methods used in neural_network.py
    

"""

import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def load_data(npz_file_name, total_portion, train_portion):
    """
    Reads the relevant data from a npz file for an energy/theta output network.
    
    Args:
        npz_file_name : address string to .npz-file containing simulation data
        total_portion : the actual amount of total data to be used
        train_portion : the amount of used data to traing the network, the other part
                            is used for evaluation between epochs (and afterwards?)
    Returns:
        training data, training labels, evaluation data, evaluation labels
    Raises:
        ValueError : if portions is not properly set
    """
    if not total_portion > 0 and total_portion <= 1:
        raise ValueError('total_portion must be in the interval (0,1]')
    if not train_portion > 0 and train_portion < 1:
        raise ValueError('train_portion must be in the interval (0,1)')
    
    print('Reading data...')
    data_set = np.load(npz_file_name)
    det_data = data_set['detector_data']
    labels = data_set['energy_labels']
    no_used_events = int(len(labels)*total_portion)
    no_train = int(train_portion*no_used_events)
    print('Done! Using', no_train, 'training events and', no_used_events-no_train, 'evaluation events.')
    return det_data[:no_train], labels[:no_train], det_data[no_train:no_used_events], labels[no_train:no_used_events]



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



def plot_predictions(prediction, labels, bins=500):
    """
    Use to plot a models predictions in similar format as previous years, i.e. 2d histograms ("lasersvärd")
    
    Args:
        prediction : use model.predict(data)
        labels : to compare with predictions
    Returns:
        figure : matplotlib.pyplot.figure.Figure
        axes : matplotlib.axes.SubplotBase (tuple x3)
        events : dict with the plotted data, see below
    Raises:ror : if data and labels is not the same length
        TypeError : if model is not a keras.Model object
    """
    if len(prediction)!=len(labels):
        raise ValueError('prediction and labels need to be of same length')
        
    events = {'predicted_energy': prediction[::,0::3].flatten(),
              'correct_energy': labels[::,0::3].flatten(), 
              
              'predicted_theta': prediction[::,1::3].flatten(),
              'correct_theta': labels[::,1::3].flatten(),
              
              'predicted_phi': np.mod(prediction[::,2::3], 2*np.pi).flatten(),
              'correct_phi': labels[::,2::3].flatten()}
    
    figure, axes = plt.subplots(1,3, figsize=(13, 5))
    image = []
    image.append(axes[0].hist2d(events['correct_energy'], events['predicted_energy'], bins=bins, norm=LogNorm(), cmax = 1001))
    image.append(axes[1].hist2d(events['correct_theta'], events['predicted_theta'], bins=bins, norm=LogNorm(), cmax = 1001))
    image.append(axes[2].hist2d(events['correct_phi'], events['predicted_phi'], bins=bins, norm=LogNorm(), cmax = 1001))
    figure.tight_layout()
    return figure, axes, events

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
