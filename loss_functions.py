""" @PETER HALLDESTAM, 2020
    
    The loss function used in neural_network.py
    
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from math import factorial
from itertools import permutations


# LOSS FUNCTION PARAMETERS
LAMBDA_ENERGY = 0.1
LAMBDA_THETA = 1
LAMBDA_PHI = 0.1

OFFSET_ENERGY = 0.1     #don't divide with 0! used in relative loss
LAMBDA_VECTOR = 1       #used in vector loss
LAMBDA_XYZ = 1          #used in cartesian loss



def loss_function_wrapper(max_mult, loss_type='mse', permutation=True, cartesian_coordinates=False):
    """
    Wrapper function for permutation_loss and non_permutation_loss. Available 
    types of loss functions is:
        mse : standard mean squared error
        modulo : implementation of last years loss function
        vector : Davids funky loss
        cosine : uses a cosine term instead of modulo for periodicity
    Args:
        max_mult : maximum multiplicity
        loss_type : what kind off loss function to use
        permutation : set True to pair with correct label permutation
        cartesian_coordinates : set True if training with cartesian coordinates
    Returns:
        custom keras loss function
    """
    if max_mult >= 7:
        print('WARNING: max multiplicity: ' + max_mult + ' is freaking huge!!')
        
    if permutation and not cartesian_coordinates:
        print('Using a permutation loss function of type ' + loss_type)
        permutation_tensor = get_permutation_tensor(max_mult)
        identity_tensor = get_identity_tensor(max_mult)
        return lambda y, y_: permutation_loss(y, y_, permutation_tensor, identity_tensor, loss_type)
    
    if not permutation and not cartesian_coordinates:
        print('Using a non-permutation loss function of type: ' + loss_type)
        print('--> Make sure you train on an ordered label set!')
        return lambda y, y_: non_permutation_loss(y, y_, loss_type)

    if cartesian_coordinates:
        print('Using a permutations loss function in cartesian coordinates of type ' + loss_type)
        permutation_tensor = get_permutation_tensor(max_mult, m=4)
        identity_tensor = get_identity_tensor(max_mult, m=4)
        return lambda y, y_: permutation_loss(y, y_, permutation_tensor, identity_tensor, loss_type, cartesian_coordinates=True)



def permutation_loss(y, y_, permutation_tensor, identity_tensor, loss_type, cartesian_coordinates=False):
    """
    Loss functions that enable training without an ordered label set. Used via
    the wrapper function permutation_loss_wrapper above. It calculates the loss
    for every possibly prediction-label pair in order to match each predicted
    event with corresponding label. The idea is the same as previous years, but
    here there's no kind of loop over each permutation. Instead, this method
    uses tensors that transforms the label matrix into max_mult! number of
    matrices for each permutation etc.

    """
    #get all possible combinations
    y_ = tf.transpose(K.dot(y_, permutation_tensor), perm=[1,0,2])
    y = tf.transpose(K.dot(y, identity_tensor), perm=[1,0,2])
    
    #what loss functions to minimize
    loss = Loss_function(loss_type, cartesian_coordinates=cartesian_coordinates).loss(y, y_)
    return K.mean(K.min(K.sum(loss, axis=2), axis=0))


def non_permutation_loss(y, y_, loss_type):
    """
    Loss functions trained with an ordered label set and does not check every
    permutation.
    
    """
    #just so these work with already defined methods
    y = K.expand_dims(y, axis=0)
    y_ = K.expand_dims(y_, axis=0)
    
    loss = Loss_function(loss_type).loss(y, y_)
    return K.mean(K.sum(loss, axis=1))





class Loss_function(object):
    def __init__(self, loss_type, cartesian_coordinates=False):
        self.loss = False
        if cartesian_coordinates:
            if loss_type=='mse': self.loss = self.cartesian_mse_loss
        
        else:
            if loss_type=='mse': self.loss = self.mse_loss
            elif loss_type=='modulo': self.loss = self.modulo_loss
            elif loss_type=='vector': self.loss = self.davids_vector_loss
            elif loss_type=='cosine': self.loss = self.cosine_loss
        
        if not self.loss: raise ValueError('invalid loss function type')
        
    ## ------------------ SPHERICAL COORDINATE LOSS FUNCTIONS ---------------------
    def mse_loss(self, y, y_):
        loss_energy = K.square(y[::,::,0::3] - y_[::,::,0::3])
        loss_theta = K.square(y[::,::,1::3] - y_[::,::,1::3])
        loss_phi = K.square(y[::,::,2::3] - y_[::,::,2::3])
        return LAMBDA_ENERGY*loss_energy + LAMBDA_THETA*loss_theta + LAMBDA_PHI*loss_phi
    
    def modulo_loss(self, y, y_):
        loss_energy = LAMBDA_ENERGY*K.square(tf.divide(y[::,::,0::3] - y_[::,::,0::3], y_[::,::,0::3] + OFFSET_ENERGY))
        loss_theta = LAMBDA_THETA*K.square(y[::,::,1::3] - y_[::,::,1::3])
        loss_phi = LAMBDA_PHI*K.square(tf.math.mod(y[::,::,2::3] - y_[::,::,2::3] + np.pi, 2*np.pi) - np.pi)       
        return loss_energy+loss_theta+loss_phi
    
    def davids_vector_loss(self, y, y_):
        x = tf.math.multiply(tf.math.sin(y[::,::,1::3]), tf.math.cos(y[::,::,2::3])) 
        y = tf.math.multiply(tf.math.sin(y[::,::,1::3]), tf.math.sin(y[::,::,2::3]))
        z = tf.math.cos(y[::,::,1::3])
        
        x_ =  tf.math.multiply(tf.math.sin(y_[::,::,1::3]), tf.math.cos(y_[::,::,2::3]))
        y_ = tf.math.multiply(tf.math.sin(y_[::,::,1::3]), tf.math.sin(y_[::,::,2::3]))
        z_ = tf.math.cos(y_[::,::,1::3])
        
        loss_energy = LAMBDA_ENERGY*K.square(y[::,::,0::3] - y_[::,::,0::3])
        loss_vector = LAMBDA_VECTOR*(1-tf.math.multiply(x,x_)-tf.math.multiply(y,y_)-tf.math.multiply(z,z_))
        return loss_energy+loss_vector
    
    def cosine_loss(self, y, y_):
        loss_energy = LAMBDA_ENERGY*K.square(y[::,::,0::3] - y_[::,::,0::3])
        loss_theta = LAMBDA_THETA*K.square(y[::,::,1::3] - y_[::,::,1::3])
        loss_phi = LAMBDA_PHI*(1-K.cos(y[::,::,2::3] - y_[::,::,2::3]))    
        return loss_energy+loss_theta+loss_phi
    
    ## ---------------- CARTESIAN COORDINATE LOSS FUNCTIONS -----------------------
    
    def cartesian_mse_loss(self, y, y_):
        loss_energy = LAMBDA_ENERGY*K.square(y[::,::,0::4]-y_[::,::,0::4])
        loss_x = K.square(y[::,::,1::4]-y_[::,::,1::4])
        loss_y = K.square(y[::,::,2::4]-y_[::,::,2::4])
        loss_z = K.square(y[::,::,3::4]-y_[::,::,3::4])
        return LAMBDA_ENERGY*loss_energy + LAMBDA_XYZ*(loss_x + loss_y + loss_z)



def get_permutation_tensor(n, m=3):
    """
    Returns a tensor containing all n! permutation matrices. Dimension: [n!, m*n, n*n]
    m is the no. outputs per reconstructed photon (4 in cartesian coordinates)
    """
    permutation_tensor = np.zeros((factorial(n), m*n, m*n))
    depth = 0
    for perm in permutations(range(n)):    
        for i in range(n):
            permutation_tensor[depth, m*i:m*(i+1):, m*perm[i]:m*(perm[i]+1):] = np.identity(m)
        depth += 1
    return K.constant(permutation_tensor)


def get_identity_tensor(n, m=3):
    """
    Returns a tensor containing n! identity matrices. Dimension: [n!, m*n, m*n]
    m is the no. outputs per reconstructed photon (4 in cartesian coordinates)
    """
    identity_tensor = np.zeros((factorial(n), m*n, m*n))
    for depth in range(factorial(n)):
            for i in range(n):
                identity_tensor[depth, m*i:m*(i+1), m*i:m*(i+1)] = np.identity(m)
    return K.constant(identity_tensor)
