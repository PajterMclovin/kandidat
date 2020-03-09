#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:39:11 2019

@author: rickardkarlsson
"""

# Importing installed modules
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import tensorflow as tf
from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse

# Needed if checking memory usage by graphic card,
# needed if the trained network had this feature
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
with tf.device('/device:GPU:0'):  # Checks memory usage
     bytes_in_use = BytesInUse()

# Importing our own modules
import conv_methods as cm
ANN_save_load = cm.module_from_file("ANN_save_load", "../../save_load/ANN_save_load.py")
fcm = cm.module_from_file("FC_methods","../../FCN/FC_methods.py")
pm = cm.module_from_file("plot_methods","../../plot_methods/plot_methods.py")
loss_methods = cm.module_from_file("loss_functions","../../loss_functions/loss_functions.py")

# Load network and parameters
save_folder = sys.argv[1]
print('Loading network from', save_folder)
list_names = ['loss_value_eval','loss_value_train']
sess, model_vars, x, y, y_ = ANN_save_load.load_model(save_folder)
data, params = ANN_save_load.load_training_results(save_folder)
loss_value_train = data['loss_value_train']
loss_value_eval = data['loss_value_eval']
multiplicity = int(params['max_multiplicity']*3)
training_portion = params['training_portion']
used_data_portion = params['used_data_portion']

# Get data to evaluate the network on
npz_data = sys.argv[2]
print("Loading data from", npz_data)
[det_data_train,
 gun_data_train,
 det_data_eval,
 gun_data_eval] = fcm.read_data_npz_energy_angle(npz_data, training_portion, used_data_portion)

#Reconstruction of the evaluation data set by using the loaded network:
print("Running network with evaluation data")
gun_data_from_network = np.empty((0,multiplicity))
for i in range(0,det_data_eval.shape[0],100):
    output_from_network = np.asarray(sess.run(y, feed_dict={x: det_data_eval[i:i+100][:]}))
    gun_data_from_network = np.concatenate((gun_data_from_network, output_from_network), axis = 0)
print("Finished")
gun_data_eval = gun_data_eval[0:gun_data_from_network.shape[0]]

# Permutation sort from the cost function
print("Sorting permutations")
events = loss_methods.network_permutation_sort(gun_data_from_network, gun_data_eval)

# Sort events in a dictionary
for key in events:
    values = np.asarray(events.get(key))
    events.update({key:values})
events.update({4: np.mod(events[4], 2*np.pi)})


max_energy_angle_values = [10, np.pi, 2*np.pi]
print("Plotting and printing results")
pm.loss_curves(loss_value_train, loss_value_eval)
pm.energy_theta_phi_splitted(events[0], events[2], events[4], events[1], events[3], events[5])

plt.show()
