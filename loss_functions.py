# made by the 2019 group

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import itertools as it

'''The standard loss function, testing all different combinations of paring of the gamma-rays.
y                           = network output
y_                          = correct labels corresponding to the network output
num_splits                  = the maximum multiplicity of the events
lam_X                       = weighting factors
E_offset                    = the energy offset used for relatice loss
theta_offset/phi_offset     = do not use, left from beginning of the project
use_relative_loss           = if relative loss is going to be used; 1 = yes and 0 = no
weight_decay_loss           = the weight decay loss (returned from the hidden layer method)
beta                        = weighting factor for weight decay loss (probably do not use; 0)
*Can be used for both relativistic (y and y_ in same frame (beam/lab)) and non relativistic data
'''
# param =(lam_E, lam_theta, lam_phi, E_offset, theta_offset, phi_offset, use_relative_loss, weight_decay_loss, beta)
# keras version of energy_theta_phi_permutation_loss (relative)
def relative_loss(y, y_, num_splits, param):
    #Dividing into individual gamma-ray blocks
    splited_y = tf.split(y, num_splits, axis=1)
    splited_y_ = tf.split(y_, num_splits, axis=1)
    temp_shape = K.shape(tf.split(splited_y[0], 3, axis=1))
    return K.zeros(temp_shape)
    
    
    def one_comb_loss(splited_y, splited_y_, index_list):
        temp = K.zeros(temp_shape, dtype=tf.float32)   # if fucked up, use tf.float32
        for i in range(len(index_list)):
            E, theta, phi = tf.split(splited_y[i], 3, axis=1)
            E_, theta_, phi_ = tf.split(splited_y_[index_list[i]], 3, axis=1)
            
            tmp_loss_E = lam_E*K.square(tf.divide(E-E_, E_+E_offset))   # if fucked, use lambda instead of tf.
            tmp_loss_theta = lam_theta*K.square(theta-theta_)
            tmp_loss_phi = lam_phi*K.square(tf.mod(phi-phi_+np.pi, 2*np.pi)-np.pi)  # may fuck up
            
            temp = temp + tmp_loss_E + tmp_loss_theta + tmp_loss_phi
        return temp
            
        #Loops over every possible pairing permutation and calculates the losses with one_comb_loss, gathered losses in list_of_tensors
        list_of_tensors = [one_comb_loss(splited_y, splited_y_, index_list) for index_list in it.permutations(range(num_splits), num_splits)]

        #Initialize infinite loss and then iterate over all the losses to find the lowest
        loss = tf.divide(K.constant(1, dtype=tf.float32), K.zeros(temp_shape, dtype=tf.float32))
        for i in range(len(list_of_tensors)):
            loss = K.min(loss, list_of_tensors[i])
        return tf.reduce_mean(loss) + beta*weight_decay_loss
    
    
#there's some other loss functions in their code, but for now I'll only use this one
def energy_theta_phi_permutation_loss(y, y_, num_splits, lam_E = 1, lam_theta = 1, lam_phi = 1, E_offset = 0.1, theta_offset = 0.1, phi_offset = 0.1, use_relative_loss = 0, weight_decay_loss = 0, beta = 0):
    #Dividing into individual gamma-ray blocks
    splited_y = tf.split(y, num_splits, axis=1)
    splited_y_ = tf.split(y_, num_splits, axis=1)
    temp_shape = tf.shape(tf.split(splited_y[0], 3, axis=1))
    return 1
    
    # #Calculates the loss for ONE of the possible permutations of the pairing
    def one_comb_loss(splited_y, splited_y_, index_list):
        temp = tf.zeros(temp_shape, dtype=tf.float32)
        if use_relative_loss == 1:
            #Relative loss
            for i in range(len(index_list)):
                E, theta, phi = tf.split(splited_y[i], 3, axis=1)
                E_, theta_, phi_ = tf.split(splited_y_[index_list[i]], 3, axis=1)

                tmp_loss_E = lam_E*tf.square(tf.divide(E-E_, E_+E_offset))
                tmp_loss_theta = lam_theta*tf.square(theta-theta_)
                tmp_loss_phi = lam_phi*tf.square(tf.mod(phi - phi_ + np.pi, 2*np.pi) - np.pi)

                temp = temp + tmp_loss_E + tmp_loss_theta + tmp_loss_phi
        else:
            #Absolute loss
            for i in range(len(index_list)):
                E, theta, phi = tf.split(splited_y[i], 3, axis=1)
                E_, theta_, phi_ = tf.split(splited_y_[index_list[i]], 3, axis=1)

                tmp_loss_E = lam_E*tf.square(E-E_)
                tmp_loss_theta = lam_theta*tf.square(theta-theta_)
                tmp_loss_phi = lam_phi*tf.square(tf.mod(phi - phi_ + np.pi, 2*np.pi) - np.pi)

                temp = temp + tmp_loss_E + tmp_loss_theta + tmp_loss_phi
        return temp

    #Loops over every possible pairing permutation and calculates the losses with one_comb_loss, gathered losses in list_of_tensors
    list_of_tensors = [one_comb_loss(splited_y, splited_y_, index_list) for index_list in it.permutations(range(num_splits), num_splits)]

    #Initialize infinite loss and then iterate over all the losses to find the lowest
    loss = tf.divide(tf.constant(1, dtype=tf.float32), tf.zeros(temp_shape, dtype=tf.float32))
    for i in range(len(list_of_tensors)):
        loss = tf.minimum(loss, list_of_tensors[i])
    return tf.reduce_mean(loss) + beta*weight_decay_loss