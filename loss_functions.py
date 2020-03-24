""" @PETER HALLDESTAM, 2020
    
    The loss function used in neural_network.py
    
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.math as m


# LOSS FUNCTION PARAMETERS
LAMBDA_ENERGY = 1
OFFSET_ENERGY = 0.1     #don't divide with 0! used in relative loss
LAMBDA_THETA = 1
LAMBDA_PHI = 1
LAMBDA_VECTOR = 1

"""
    OBS. why calculate the loss for all permutations? Why not train the network to
    predict the events ordered in decreasing energy??????? 
    
"""

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
    energy, energy_ = y[::,0::3], y_[::,0::3]
    theta, theta_ = y[::,1::3], y_[::,1::3]
    phi, phi_ = y[::,2::3], y[::,2::3]
    
    loss_energy = LAMBDA_ENERGY*K.square(tf.divide(energy-energy_, energy_+OFFSET_ENERGY))
    loss_theta = LAMBDA_THETA*K.square(theta-theta_)
    loss_phi = LAMBDA_PHI*K.square(m.mod(phi - phi_ + np.pi, 2*np.pi) - np.pi)
    return K.mean(loss_energy + loss_theta + loss_phi)


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
    phi, phi_ = y[::,2::3], y[::,2::3]
    
    loss_energy = LAMBDA_ENERGY*K.square(energy-energy_)
    loss_theta = LAMBDA_THETA*K.square(theta-theta_)
    loss_phi = LAMBDA_PHI*K.square(m.mod(phi - phi_ + np.pi, 2*np.pi) - np.pi)
    return K.mean(loss_energy + loss_theta + loss_phi)
    
def vector_loss(y, y_):
     
     """ 
     vektorbaserad kostnadsfunktion, förmodar att indata är i samma format som 2019 (energi, vinklar)
     """
     
     energy, energy_ = y[::,0::3], y_[::,0::3]
     theta, theta_ = y[::,1::3], y_[::,1::3]
     phi, phi_ = y[::,2::3], y[::,2::3]
     
     x,x_ = m.multiply(m.sin(theta), m.cos(phi)), m.multiply(m.sin(theta_), m.cos(phi_))
     y,y_ = m.multiply(m.sin(theta), m.sin(phi)), m.multiply(m.sin(theta_), m.sin(phi_))
     z,z_ = m.cos(theta), m.cos(theta_)
     
     r = [x,y,z]
     r_ = [x_,y_,z_]
     
     loss_energy = LAMBDA_ENERGY*K.square(energy-energy_)
     loss_angles = LAMBDA_VECTOR*(np.ones(r.shape)-K.dot(r,r_))
     
     return K.mean(loss_energy + loss_angles)

