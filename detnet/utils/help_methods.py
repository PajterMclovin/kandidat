""" @PETER HALLDESTAM, 2020

    Help methods used in neural_network.py
    
"""
import os
import numpy as np

from tensorflow import transpose
from tensorflow.keras import Model

from tensorflow.keras.backend import dot
from tensorflow.keras.backend import constant
from tensorflow.keras.backend import count_params
from tensorflow.keras.backend import argmin
from tensorflow.keras.backend import sum as Ksum

from utils.tensors import get_permutation_tensor
from utils.tensors import get_identity_tensor

DET_GEOMETRY = os.path.join(os.getcwd(), 'data', 'geom_xb.txt') 

def get_permutation_match(y, y_, loss_function, max_mult, no_batches=10):
    """
    Sorts the predictions with corresponding label as the minimum of a square 
    error loss function. Must be used BEFORE plotting the "lasersvärd".

    """
    print('Matching predicted data with correct label permutation. May take a while...')
    
    new_y_ = np.empty((0, len(y[0])))
    loss = loss_function.get(train=False)
    
    no_params = int(len(y_[0])/max_mult)
    permutation_tensor = get_permutation_tensor(max_mult, m=no_params)
    

    prediction_batches = np.array_split(y, no_batches)
    labels_batches = np.array_split(y_, no_batches)
    
    for Y, Y_ in zip(prediction_batches, labels_batches):
        
        tY, tY_ = constant(Y), constant(Y_)
        indices = np.array(argmin(loss(tY, tY_), axis=0))
        tY_ = np.array(transpose(dot(tY_, permutation_tensor), perm=[1,0,2]))
        matched_y_ = [tY_[j, i, ::] for i,j in enumerate(indices)]
        new_y_ = np.append(new_y_, matched_y_, axis=0)
        
    
    return y, new_y_


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
    with open(DET_GEOMETRY) as f:
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
    assert isinstance(compiled_model, Model)
    return np.sum([count_params(w) for w in compiled_model.trainable_weights])


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



