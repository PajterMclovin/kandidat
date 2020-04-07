""" @PETER HALLDESTAM, 2020
    
    The loss function used in neural_network.py
    
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from transformations import get_identity_tensor
from transformations import get_permutation_tensor
from transformations import get_shift_tensor

# LOSS FUNCTION PARAMETERS
LAMBDA_ENERGY = 1
LAMBDA_THETA = 1
LAMBDA_PHI = 1

OFFSET_ENERGY = 0.1     #don't divide with 0! used in relative loss
LAMBDA_VECTOR = 10       #used in vector loss
LAMBDA_XYZ = 1          #used in cartesian loss
LAMBDA_CLASSIFICATION = 1 #used in binary cross entropy


def loss_function_wrapper(no_outputs, loss_type='mse', permutation=True, 
                          cartesian=False, classification=False):
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
        classification_nodes : set True if training with classification nodes
    Returns:
        custom keras loss function
    """    
    shift_tensor = False
    no_param = 3
    
    if cartesian:
        print('Training with cartesian coordinates (px, py, pz)')
    else:
        print('Training with spherical coordinates (energy, theta, phi)')
        
    if classification:
        print('Training with classification nodes')
        no_param += 1   #mu
        shift_tensor = get_shift_tensor(int(no_outputs/no_param), m=no_param)
    
    max_mult = int(no_outputs/no_param)
    if max_mult >= 7:
        print('WARNING: max multiplicity: ' + max_mult + ' is freaking huge!!')
        
    if permutation:
        print('Using a permutation loss function of type ' + loss_type)
        permutation_tensor = get_permutation_tensor(max_mult, m=no_param)
        identity_tensor = get_identity_tensor(max_mult, m=no_param)
        return lambda y, y_: permutation_loss(y, y_, loss_type, 
                                              permutation_tensor, 
                                              identity_tensor, 
                                              cartesian=cartesian,
                                              classification=classification,
                                              shift_tensor=shift_tensor)
    
    if not permutation:
        print('Using a non-permutation loss function of type: ' + loss_type)
        print('--> Make sure you train on an ordered label set!')
        return lambda y, y_: non_permutation_loss(y, y_, loss_type, 
                                                  cartesian=cartesian,
                                                  classification=classification)


def permutation_loss(y, y_, loss_type, permutation_tensor, identity_tensor,
                     cartesian=False, classification=False, shift_tensor=False):
    """
    Loss functions that enable training without an ordered label set. Used via
    the wrapper function permutation_loss_wrapper above. It calculates the loss
    for every possibly prediction-label pair in order to match each predicted
    event with corresponding label. The idea is the same as previous years, but
    here there's no kind of loop over each permutation. Instead, this method
    uses tensors that transforms the label matrix into max_mult! number of
    matrices for each permutation etc.

    """
    
    if classification:
        y_ = K.dot(y, shift_tensor)
    
    #get all possible combinations
    y_ = tf.transpose(K.dot(y_, permutation_tensor), perm=[1,0,2])
    y = tf.transpose(K.dot(y, identity_tensor), perm=[1,0,2])
    
    #what loss functions to minimize
    loss = Loss_function(loss_type, cartesian=cartesian,classification=classification).loss(y, y_)
    return K.mean(K.min(K.sum(loss, axis=2), axis=0))


def non_permutation_loss(y, y_, loss_type, cartesian=False,
                     classification=False, shift_tensor=False):
    """
    Loss functions trained with an ordered label set and does not check every
    permutation.
    
    """
    #just so these work with already defined methods
    y = K.expand_dims(y, axis=0)
    y_ = K.expand_dims(y_, axis=0)
    
    loss = Loss_function(loss_type, cartesian=cartesian, classification=classification).loss(y, y_)
    return K.mean(K.sum(loss, axis=1))





class Loss_function(object):
    def __init__(self, loss_type, cartesian=False, classification=False):
        self.loss = False
        self.loss_type = loss_type
        self.cartesian = cartesian
        
        if classification:
            self.loss = self.loss_with_binary_cross_entropy
        else:
            self.loss = self.get_loss_type()
        if not self.loss:
            raise ValueError('invalid loss function type')
            
    def get_loss_type(self):
        if self.cartesian:
            return self.momentum_loss
        else:
            if self.loss_type=='mse':  return self.mse_loss
            elif self.loss_type=='modulo': return self.modulo_loss
            elif self.loss_type=='cosine': return self.cosine_loss
            
            
    ## ------------------ SPHERICAL COORDINATE LOSS FUNCTIONS ---------------------
    def mse_loss(self, y, y_):
        loss_energy = K.square(tf.divide(y[::,::,0::3] - y_[::,::,0::3], y_[::,::,0::3] + OFFSET_ENERGY))
        loss_theta = K.square(y[::,::,1::3] - y_[::,::,1::3])
        loss_phi = K.square(y[::,::,2::3] - y_[::,::,2::3])
        return LAMBDA_ENERGY*loss_energy + LAMBDA_THETA*loss_theta + LAMBDA_PHI*loss_phi
    
    def modulo_loss(self, y, y_):
        loss_energy = LAMBDA_ENERGY*K.square(tf.divide(y[::,::,0::3] - y_[::,::,0::3], y_[::,::,0::3] + OFFSET_ENERGY))
        loss_theta = LAMBDA_THETA*K.square(y[::,::,1::3] - y_[::,::,1::3])
        loss_phi = LAMBDA_PHI*K.square(tf.math.mod(y[::,::,2::3] - y_[::,::,2::3] + np.pi, 2*np.pi) - np.pi)       
        return loss_energy+loss_theta+loss_phi

    
    def cosine_loss(self, y, y_):
        loss_energy = LAMBDA_ENERGY*K.square(y[::,::,0::3] - y_[::,::,0::3])
        loss_theta = LAMBDA_THETA*K.square(y[::,::,1::3] - y_[::,::,1::3])
        loss_phi = LAMBDA_PHI*(1-K.cos(y[::,::,2::3] - y_[::,::,2::3]))    
        return loss_energy+loss_theta+loss_phi
    
    ## --------------- CARTESIAN COORDINATE LOSS FUNCTIONS --------------------
    
    def momentum_loss(self, y, y_):
        px, px_ = y[...,0::3], y_[...,0::3]
        py, py_ = y[...,1::3], y_[...,1::3]
        pz, pz_ = y[...,2::3], y_[...,2::3]
        return K.square(px-px_) + K.square(py-py_) + K.square(pz-pz_)


    ## ----------------- BINARY CLASSIFICATION LOSS FUNCTIONS -----------------
    
    # def split_regress_class(self, y, y_):
        
    #     if self.cartesian_coordinates:
    #         mu, mu_ = y[::,::,0::5], y_[::,::,0::5]
    #         energy, energy_ = y[::,::,1::5], y_[::,::,1::5]
    #         X, X_ = y[::,::,2::5], y_[::,::,2::5]
    #         Y, Y_ = y[::,::,3::5], y_[::,::,3::5]
    #         Z, Z_ = y[::,::,4::5], y_[::,::,4::5]
            
    #     loss_binary_cross_entropy = -LAMBDA_CLASSIFICATION*(mu_*K.log(mu)+(1-mu_)*K.log(1-mu))
        
    #     loss_regular = self.get_loss_type()(y, y_)
    #     return loss_regular+loss_binary_cross_entropy
        
        
