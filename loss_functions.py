""" @PETER HALLDESTAM, 2020
    
    The loss function used in neural_network.py
    
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from utils import get_permutation_tensor, get_identity_tensor

# LOSS FUNCTION PARAMETERS
LAMBDA_ENERGY = 1
LAMBDA_THETA = 1
LAMBDA_PHI = 1

OFFSET_ENERGY = 0.1     #don't divide with 0! used in relative loss
LAMBDA_VECTOR = 1       #used in vector loss


def permutation_loss_wrapper(max_mult, loss_type='mean_squared_error'):
    """
    Wrapper function for permutation_loss. Types of loss functions is:
        mean_squared_error : standard
        modulo : implementation of last years loss function
        vector : Davids funky loss
        cosine : uses a cosine term instead of modulo for periodicity
    Args:
        max_mult : maximum multiplicity
        loss_type : what kind off loss function to use
    
    """
    if max_mult >= 7:
        print('WARNING: max multiplicity: ' + max_mult + ' is freaking huge!!')
    permutation_tensor = get_permutation_tensor(max_mult)
    identity_tensor = get_identity_tensor(max_mult)
    def loss(y, y_):
        return permutation_loss(y, y_, permutation_tensor, identity_tensor, loss_type)
    return loss


def permutation_loss(y, y_, permutation_tensor, identity_tensor, loss_type):
    """
    A loss function that enables training without orderes labels (in fact, it 
    requires no such clear order, e.g. decreasing energy). Used via the
    wrapper function permutation_loss_wrapper above.
    
    """
    y_ = tf.transpose(K.dot(y_, permutation_tensor), perm=[1,0,2])
    y = tf.transpose(K.dot(y, identity_tensor), perm=[1,0,2])
    
    energy, energy_ = y[::,::,0::3], y_[::,::,0::3]
    theta, theta_ = y[::,::,1::3], y_[::,::,1::3]
    phi, phi_ = y[::,::,2::3], y_[::,::,2::3]
    
    if loss_type=='mean_squared_error':
        loss_energy = LAMBDA_ENERGY*K.square(energy - energy_)
        loss_theta = LAMBDA_THETA*K.square(theta - theta_)
        loss_phi = LAMBDA_PHI*K.square(phi - phi_)
        
    elif loss_type=='modulo':
        loss_energy = LAMBDA_ENERGY*K.square(tf.divide(energy - energy_, energy_ + OFFSET_ENERGY))
        loss_theta = LAMBDA_THETA*K.square(theta-theta_)
        loss_phi = LAMBDA_PHI*K.square(tf.math.mod(phi - phi_ + np.pi, 2*np.pi) - np.pi)        
        
    elif loss_type=='vector':
        x,x_ = tf.math.multiply(tf.math.sin(theta), tf.math.cos(phi)), tf.math.multiply(tf.math.sin(theta_), tf.math.cos(phi_))
        y,y_ = tf.math.multiply(tf.math.sin(theta), tf.math.sin(phi)), tf.math.multiply(tf.math.sin(theta_), tf.math.sin(phi_))
        z,z_ = tf.math.cos(theta), tf.math.cos(theta_)

        loss_energy = LAMBDA_ENERGY*K.square(energy-energy_)
        loss_theta = LAMBDA_VECTOR*(1-tf.math.multiply(x,x_)-tf.math.multiply(y,y_)-tf.math.multiply(z,z_))
        loss_phi = 0
        
    elif loss_type=='cosine':
        loss_energy = LAMBDA_ENERGY*K.square(energy - energy_)
        loss_theta = LAMBDA_THETA*(1-K.cos(theta-theta_))
        loss_phi = LAMBDA_PHI*(1-K.cos(phi-phi_))
        
    
    return K.mean(K.min(K.sum(loss_energy+loss_theta+loss_phi, axis=2), axis=0))



def relative_loss(y, y_):
    """
    tensorflow.kereas.backend symbolic loss function with relative error in energy.
    Used in model.fit
    
    Args:
        y : predicted data from network
        y_ : corresponding labels
    Returns:
        loss
    """
    energy, energy_ = y[::,0::], y_[::,0::3]
    theta, theta_ = y[::,1::3], y_[::,1::3]                 #zenith (0,pi)
    phi, phi_ = y[::,2::3], y_[::,2::3]                      #azimuth (0,2pi)
    
    # loss_energy = K.mean(K.square(tf.divide(energy-energy_, energy_+OFFSET_ENERGY)))
    
    loss_energy = K.mean(K.square(energy-energy_))
    # loss_theta = K.mean(K.square(theta-theta_))
    # loss_phi = K.mean(K.square(phi-phi_))
    
    # loss_theta = K.mean(tf.abs(tf.atan2(tf.sin(theta-theta_), tf.cos(theta-theta_))))
    # loss_phi = K.mean(tf.abs(tf.atan2(tf.sin(phi-phi_), tf.cos(phi-phi_))))
    
    loss_theta = K.mean(1-K.cos(theta-theta_))
    loss_phi = K.mean(1-K.cos(phi-phi_))
    return LAMBDA_ENERGY*loss_energy + LAMBDA_THETA*loss_theta + LAMBDA_PHI*loss_phi


def absolute_loss(y, y_):
    """
    tensorflow.kereas.backend symbolic loss function with absolute error in energy.
    Used in model.fit
    
    Args:
        y : predicted data from network
        y_ : corresponding labels
    Returns:
        loss
    """
    energy, energy_ = y[::,0::3], y_[::,0::3]
    theta, theta_ = y[::,1::3], y_[::,1::3]
    phi, phi_ = y[::,2::3], y_[::,2::3]
    
    loss_energy = LAMBDA_ENERGY*K.square(energy-energy_)
    loss_theta = LAMBDA_THETA*K.square(theta-theta_)
    loss_phi = LAMBDA_PHI*K.square(tf.math.mod(phi - phi_ + np.pi, 2*np.pi) - np.pi)
    return K.mean(loss_energy + loss_theta + loss_phi)

### ---------- Davids code ----------------------------------------------------      

def vector_loss(y, y_):
     
     """ 
     vektorbaserad kostnadsfunktion, förmodar att indata är i samma format som 2019 (energi, vinklar)
     """

     
     energy, energy_ = y[::,0::3], y_[::,0::3]
     theta, theta_ = y[::,1::3], y_[::,1::3]
     phi, phi_ = y[::,2::3], y_[::,2::3]
     
     x,x_ = tf.math.multiply(tf.math.sin(theta), tf.math.cos(phi)), tf.math.multiply(tf.math.sin(theta_), tf.math.cos(phi_))
     y,y_ = tf.math.multiply(tf.math.sin(theta), tf.math.sin(phi)), tf.math.multiply(tf.math.sin(theta_), tf.math.sin(phi_))
     z,z_ = tf.math.cos(theta), tf.math.cos(theta_)

     loss_energy = LAMBDA_ENERGY*K.square(energy-energy_)
     loss_angles = LAMBDA_VECTOR*(1-tf.math.multiply(x,x_)-tf.math.multiply(y,y_)-tf.math.multiply(z,z_))
     
     return K.mean(loss_energy + loss_angles)

def vector_loss_cart(u,u_):
    """
    
    vektorbaserad kostnadsfuntion, indata redan i kartesiska koordinater
    """
    
    energy, energy_ = u[::,0::4], u_[::,0::4]
    x, x_ = u[::,1::4], u_[::,1::4]
    y, y_ = u[::,2::4], u_[::,2::4]
    z, z_ = u[::,3::4], u_[::,3::4]
    
    dot = x*x_ + y*y_ + z*z_
    norm = K.sqrt(x*x + y*y + z*z)
    
    loss_energy = LAMBDA_ENERGY*K.square(energy-energy_)
    loss_spacial = LAMBDA_VECTOR*K.square(1-dot/norm)
    
    return K.mean(loss_energy + loss_spacial)

def mean_square_cart(u,u_):
    
    energy, energy_ = u[::,0::4], u_[::,0::4]
    x, x_ = u[::,1::4], u_[::,1::4]
    y, y_ = u[::,2::4], u_[::,2::4]
    z, z_ = u[::,3::4], u_[::,3::4]
    
    loss_energy = LAMBDA_ENERGY*K.square(energy-energy_)
    loss_x = LAMBDA_VECTOR*K.square(x-x_)
    loss_y = LAMBDA_VECTOR*K.square(y-y_)
    loss_z = LAMBDA_VECTOR*K.square(z-z_)
    return K.mean(loss_energy + loss_x + loss_y + loss_z)
