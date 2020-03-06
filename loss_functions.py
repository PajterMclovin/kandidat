""" @PETER HALLDESTAM, 2020
    
    The loss function used in neural_network.py, which builds upon the 2019 group's code.  
    
"""

import numpy as np
import tensorflow as tf
# import tensorflow.keras.backend as K
import itertools as it

class Permutation_loss(tf.keras.losses.Loss):
    """
    Args:
        num_splits                  : the maximum multiplicity of the events
        lam_X                       : weighting factors
        E_offset                    : the energy offset used for relatice loss
        theta_offset/phi_offset     : do not use, left from beginning of the project
        use_relative_loss           : if relative loss is going to be used; 1 = yes and 0 = no
        weight_decay_loss           : the weight decay loss (returned from the hidden layer method)
        beta                        : weighting factor for weight decay loss (probably do not use; 0) 
    """
    def __init__(self, loss_name, num_splits, lam_E, lam_theta, lam_phi,
                 E_offset, theta_offset, phi_offset,
                 weight_decay_loss, beta):
        super().__init__(name=loss_name)        #reduction=tf.keras.losses.Reduction.AUTO)
        self.num_splits = num_splits
        self.lam_E = lam_E
        self.lam_theta = lam_theta
        self.lam_phi = lam_phi
        self.E_offset = E_offset
        self.theta_offset = theta_offset
        self.phi_offset = phi_offset
        self.weight_decay_loss = weight_decay_loss
        self.beta = beta
    
    def one_comb_loss(self, splited_y, splited_y_, index_list, shape):
        pass
    
    def call(self, y, y_):
        #Dividing into individual gamma-ray blocks
        splited_y = tf.split(y, self.num_splits, axis=1)
        splited_y_ = tf.split(y_, self.num_splits, axis=1)
        tmp_shape = tf.shape(tf.split(splited_y[0], 3, axis=1))
        
        #Loops over every possible pairing permutation and calculates the losses with one_comb_loss, gathered losses in list_of_tensors
        list_of_tensors = [self.one_comb_loss(splited_y, splited_y_, index_list, tmp_shape) 
                           for index_list in it.permutations(range(self.num_splits), self.num_splits)]
    
        #Initialize infinite loss and then iterate over all the losses to find the lowest
        loss = tf.divide(tf.constant(1, dtype=tf.float32), tf.zeros(tmp_shape, dtype=tf.float32))
        for i in range(len(list_of_tensors)):
            loss = tf.minimum(loss, list_of_tensors[i])
    
        return tf.reduce_mean(loss) + self.beta*self.weight_decay_loss
        
    
class Relative_loss(Permutation_loss):
    """Implements the relative loss function.\n""" + Permutation_loss.__doc__
            
    def __init__(self, num_splits, lam_E, lam_theta, lam_phi,
                 E_offset, theta_offset, phi_offset,
                 weight_decay_loss, beta):
        super().__init__('relative_loss', num_splits,
                         lam_E, lam_theta, lam_phi,
                         E_offset, theta_offset, phi_offset,
                         weight_decay_loss, beta)
        
    def one_comb_loss(self, splited_y, splited_y_, index_list, shape):
        tmp = tf.zeros(shape)
        for i in range(len(index_list)):
            E, theta, phi = tf.split(splited_y[i], 3, axis=1)
            E_, theta_, phi_ = tf.split(splited_y_[index_list[i]], 3, axis=1)
            
            tmp_loss_E = self.lam_E*tf.square(tf.divide(E-E_, E_+self.E_offset))
            tmp_loss_theta = self.lam_theta*tf.square(theta-theta_)
            tmp_loss_phi = self.lam_phi*tf.square(tf.math.mod(phi-phi_+np.pi, 2*np.pi)-np.pi)
            
            tmp = tmp + tmp_loss_E + tmp_loss_theta + tmp_loss_phi
        return tmp

    

class Absolute_loss(Permutation_loss)  :
    """Implements the absolute loss function.\n""" + Permutation_loss.__doc__
    
    def __init__(self, num_splits, lam_E, lam_theta, lam_phi,
                 E_offset, theta_offset, phi_offset,
                 weight_decay_loss, beta):
        super().__init__('absolute_loss', num_splits,
                         lam_E, lam_theta, lam_phi,
                         E_offset, theta_offset, phi_offset,
                         weight_decay_loss, beta)
        
    def one_comb_loss(self, splited_y, splited_y_, index_list, shape):
        tmp = tf.zeros(shape)
        for i in range(len(index_list)):
            E, theta, phi = tf.split(splited_y[i], 3, axis=1)
            E_, theta_, phi_ = tf.split(splited_y_[index_list[i]], 3, axis=1)

            tmp_loss_E = self.lam_E*tf.square(E-E_)
            tmp_loss_theta = self.lam_theta*tf.square(theta-theta_)
            tmp_loss_phi = self.lam_phi*tf.square(tf.mod(phi - phi_ + np.pi, 2*np.pi) - np.pi)
            
            tmp = tmp + tmp_loss_E + tmp_loss_theta + tmp_loss_phi
        return tmp
    
