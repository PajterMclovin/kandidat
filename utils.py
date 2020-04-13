""" @PETER HALLDESTAM, 2020

    Help methods used in neural_network.py
    
"""
import os
import numpy as np
import tensorflow as tf

from random import shuffle
from tensorflow import transpose
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

from loss_functions import Loss_function
from loss_functions import loss_function_wrapper

from transformations import get_identity_tensor
from transformations import get_permutation_tensor

from models import FCN, GCN, CNN


def save(folder, figure, learning_curve, model):
    folder0 = folder
    folder_name_taken = True
    n = 0
    while folder_name_taken:
        n += 1
        try:
            os.makedirs(folder)
            folder_name_taken = False
        except FileExistsError:
            folder = folder0 + str(n)
        if n==20: 
            raise ValueError('change name!')
    folder = folder+'/'
    figure.savefig(folder + 'event_reconstruction.png', format='png')
    # figure.savefig(folder + 'event_reconstruction.eps', format='eps')
    learning_curve.savefig(folder + 'training_curve.png', format='png')
    model.save_weights(folder + 'weights.h5')


def load_data(npz_file_name, total_portion, 
              cartesian=False, classification=False):
    """
    Reads a .npz-file containing simulation data in spherical coordinates and
    returns data/labels in spherical or cartesian coordinates in numpy arrays
    
    Args:
        npz_file_name : address string to .npz-file
        total_portion : the actual amount of total data to be used
        cartesian_coordinates : set True for cartesian coordinates
        classification_nodes : set True to train with classification nodes
    Returns:
        all data and labels in spherical (or cartesian) coordinates
    Raises:
        ValueError : if portion is not properly set
    """
    if not total_portion > 0 and total_portion <= 1:
        raise ValueError('total_portion must be in the interval (0,1].')
    data_set = np.load(npz_file_name)
    det_data = data_set['detector_data']
    labels = data_set['energy_labels']
    if cartesian:
        labels = spherical_to_cartesian(labels)
    if classification:
        labels = insert_classification_labels(labels, cartesian=cartesian)
    no_events = int(len(labels)*total_portion)
    print('Using {} events from {}'.format(no_events, npz_file_name))
    return det_data[:no_events], labels[:no_events]


def insert_classification_labels(labels, cartesian=False):
    """
    Inserts binary classification labels for each event as:
        energy==0 => mu=1
        energy >0 => mu=0
    """
    m = 3 + cartesian
    energy = labels[::,0::m]
    pos = labels[::,[i for i in range(len(labels[0])) if np.mod(i,m)!=0]]
    mu = (energy!=0)*1
    print(len(mu))
    max_mult = int(len(labels[0])/m)
    new_labels = np.zeros((len(labels), max_mult*(m+1)))
    new_labels[::,0::m+1] = mu
    new_labels[::,1::m+1] = energy
    new_labels[::,[i for i in range(len(labels[0])+max_mult) if np.mod(i,m+1)!=0 and np.mod(i-1,m+1)!=0]]=pos
    return new_labels
    
    
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
    


def get_permutation_match(y, y_, cartesian, loss_type='mse'):
    """
    Sorts the predictions with corresponding label as the minimum of a square 
    error loss function. Must be used BEFORE plotting the "lasersvärd".

    """
    max_mult = int(len(y_[0])/3)
    permutation_tensor = get_permutation_tensor(max_mult, m=3)
    identity_tensor = get_identity_tensor(max_mult, m=3)
    
    #get all possible combinations
    Y_ = transpose(K.dot(K.variable(y_), permutation_tensor), perm=[1,0,2])
    Y = transpose(K.dot(K.variable(y), identity_tensor), perm=[1,0,2])
    
    loss_function = Loss_function(loss_type, cartesian=cartesian)
    permutation_indices = K.argmin(K.sum(loss_function.loss(Y, Y_), axis=2), axis=0)
    
    print('Matching predicted data with correct label permutation. May take a while...')
    for i in range(len(y_)):
        y_[i,::] = Y_[permutation_indices[i],i,::]    #optimize this...
    return y, y_


def spherical_to_cartesian(spherical):
    """
    Coordinate transform (energy, theta, phi) --> (px, py, pz)
    
    """
    energy = spherical[::,0::3]
    theta = spherical[::,1::3]
    phi = spherical[::,2::3]

    px = np.sin(theta)*np.cos(phi)*energy    
    py = np.sin(theta)*np.sin(phi)*energy
    pz = np.cos(theta)*energy
    
    cartesian = np.zeros(np.shape(spherical))
    cartesian[::,0::3] = px
    cartesian[::,1::3] = py
    cartesian[::,2::3] = pz
    return cartesian

def cartesian_to_spherical(cartesian, error=False):
    """
    Coordinate transform (px, py, pz) --> (energy, theta, phi). Used for labels 
    and predictions after training.

    """
    px = cartesian[::,0::3]
    py = cartesian[::,1::3]
    pz = cartesian[::,2::3]
    energy = np.sqrt(px*px + py*py + pz*pz)
    
    tol = 1e-3
    get_theta = lambda z,r: np.arccos(np.divide(z, r, out=np.ones_like(z), where=r>tol))
    get_phi = lambda y,x: np.arctan2(y,x)
    
    if error:
        zero_to_random = 0
    else:
        zero_to_random = np.random.uniform(low=-1.0, high=-.5, size=np.shape(energy))
    
    theta = np.where(energy <tol , 0, get_theta(pz, energy))
    phi = np.where(energy <tol , 0, get_phi(py, px))
    energy = np.where(energy <tol , zero_to_random, energy)
    
    spherical = np.zeros(np.shape(cartesian))
    spherical[::,0::3] = energy
    spherical[::,1::3] = theta
    spherical[::,2::3] = np.mod(phi, 2*np.pi)
    return spherical

def get_detector_angles():
    """
    Returns the angles (theta, phi) for each of 162 crystall detectors.
    
    """
    with open('geom_xb.txt') as f:
        lines = f.readlines()
        
    theta, phi = np.zeros((162,)), np.zeros((162,))
    lines = [line.strip() for line in lines]
    for i in range(162):
        s = lines[i].split(',')
        theta[i] = float(s[2])
        phi[i] =  float(s[3])
    return theta*np.pi/180, (phi+180)*np.pi/180
    

def get_no_trainable_parameters(compiled_model):
    """
    Returns the no. trainable parameters of given compiled model.
    
    """
    assert isinstance(compiled_model, tf.keras.Model)
    return np.sum([K.count_params(w) for w in compiled_model.trainable_weights])


def get_measurement_of_performance(y, y_):
    """
    Returns the mean and standard deviation in the error of predicted energy, 
    theta and phi for given predictions and labels.
    
    """
    energy_error = y[...,0::3]-y_[...,0::3]
    theta_error = y[...,1::3]-y_[...,1::3]
    phi_diff = np.mod(y[...,2::3]-y_[...,2::3], 2*np.pi)
    phi_error = np.where(phi_diff > np.pi, phi_diff - 2*np.pi, phi_diff)
    
    mean = (np.mean(energy_error), np.mean(theta_error), np.mean(phi_error))
    std = (np.std(energy_error), np.std(theta_error), np.std(phi_error))
    return {'mean': mean, 'std': std}
    



def get_trained_model(train_data, train_labels,
                    validation_split=0.1,
                    batch_size=2**8,
                    learning_rate=1e-4,
                    permutation=True,
                    cartesian=True,
                    classification=False,
                    loss_function='mse',
                    no_epochs=2,
                    depth=10, width=128):
    
    #get no. inputs and outputs in the model
    no_inputs = len(train_data[0])                  
    no_outputs = len(train_labels[0])

    #structure initialization the fully connected neural network
    model = FCN(no_inputs, no_outputs, depth, width,
                cartesian=cartesian,
                classification=classification)
    
    #select loss function
    loss = loss_function_wrapper(no_outputs, 
                                 loss_type=loss_function, 
                                 permutation=permutation,
                                 cartesian=cartesian,
                                 classification=classification)
    
    #select optimizer
    opt = Adam(lr=learning_rate)
    
    #compile the network
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    
    #train the network with training data
    training = model.fit(train_data, train_labels, 
                         epochs=no_epochs, 
                         batch_size=batch_size,
                         validation_split=validation_split)

    return model, training

def get_trained_model_conv(train_data, train_labels,
                    validation_split=0.1,
                    batch_size=2**8,
                    learning_rate=1e-4,
                    permutation=True,
                    cartesian=True,
                    classification=False,
                    loss_function='mse',
                    no_epochs=2,
                    depth=10, width=128):
    
    #get no. inputs and outputs in the model
    no_inputs = len(train_data[0])                  
    no_outputs = len(train_labels[0])

    #structure initialization the fully connected neural network
    model = CNN(no_inputs, no_outputs)
    
    #select loss function
    loss = loss_function_wrapper(no_outputs, 
                                 loss_type=loss_function, 
                                 permutation=permutation,
                                 cartesian=cartesian,
                                 classification=classification)
    
    #select optimizer
    opt = Adam(lr=learning_rate)
    
    #compile the network
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    
    #train the network with training data
    training = model.fit(train_data, train_labels, 
                         epochs=no_epochs, 
                         batch_size=batch_size,
                         validation_split=validation_split)

    return model, training

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
