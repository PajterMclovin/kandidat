#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:48:37 2019

@author: rickardkarlsson
"""

# Installed python modules
import numpy as np
import tensorflow as tf
import time
import sys
import datetime
import importlib
import matplotlib.pyplot as plt
#from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse

def pool_layer(pool_type, output_data, pool_size, pool_stride=1, padding = 'VALID'):
    if pool_type == 'avg':
        return tf.nn.avg_pool(output_data, [1, 1, pool_size, 1], [1,1,pool_stride,1], padding = padding)
    else:
        return tf.nn.max_pool(output_data, [1, 1, pool_size, 1], [1,1,pool_stride,1], padding = padding)



#Rickard's method for importing methods from other directories
def module_from_file(module_name, file_path):
    """
        Used to import modules from different folder in our Github repo.

        module_name is the file name (Ex. 'file.py')
        file_path is the file path (Ex. '../save_load/file.py) if you want to reach a
        folder in parent directory

        Outputs the module as variable that you use as an usual import reference
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Importing our own modules
#import conv_methods as cm
cm = module_from_file("conv_methods", "../../CNN/Network/conv_methods.py")
ANN_save_load = module_from_file("ANN_save_load", "../../save_load/ANN_save_load.py")
fcm = module_from_file("FC_methods","../../FCN/FC_methods.py")
pm = module_from_file("plot_methods","../../plot_methods/plot_methods.py")
loss_methods=module_from_file("loss_functions","../../loss_functions/loss_functions.py")

# print starting time
print('imports loaded at:', datetime.datetime.now())

## GPU code (comment if you don't use GPU)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


#### Memory tracing
"""
with tf.device('/device:GPU:0'):  # Checks memory usage
  bytes_in_use = BytesInUse()

run_metadata = tf.RunMetadata()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
"""

#### Network parameters
iterations = int(5e5)
batch_size = 100
batch_size_eval = 100
learning_rate = 0.0001
used_data_portion = 1
training_portion = 0.95
#text = input("Beskriv test:")
pool_size = 3
padding = 'VALID'
pool_type = 'avg'
pool_stride = 1
message = str(sys.argv) + "\n" + 'Beskrivning av testet'


#### Terminal arguments
npz_file_name = sys.argv[1]
save_folder = sys.argv[2]
if len(sys.argv) > 3:
    lossfkn = sys.argv[3]
else:
    lossfkn = None

# print starting time
print('starting training at:', datetime.datetime.now())

# Get training and evaluation data
print("Reading and preparing data")
[det_data_train,
 gun_data_train,
 det_data_eval,
 gun_data_eval] = fcm.read_data_npz_energy_angle(npz_file_name,training_portion,
                                                    used_data_portion)

# Decide mulitplicity from input data
multiplicity = np.shape(gun_data_eval)[1]

# print run-arguments to output
print(message)
print("Building the network")

# Create placeholder for input and correct values
x = tf.placeholder(dtype=tf.float32,shape=(None,cm.input_size), name = 'x')
y_ = tf.placeholder(dtype=tf.float32, shape=[None,multiplicity], name = 'y_')

# Convolutional layers

"""
Vill man använda conv_A_rotated eller conv_D_rotated prata
med Rickard om hur det implementeras
"""
# Type A crystal ###################
conv_A = np.load('conv_mat_A_with_middle.npy')
conv_A_rotated = np.load('conv_mat_A_rotated.npy') # for rotational symmetry

no_filters_per_layer = [10,10,10,10]
filter_size = 4
stride = 1
output_data_A = cm.hidden_conv_layers_without_conv_mat(x,
                                                        stride, filter_size,
                                                        no_filters_per_layer,
                                                       'A',
                                                       c_matrix= conv_A)


# Pooling of A (input, pooling size, strides, VALID means that it doesnt use padding)
output_data_A = pool_layer(pool_type, output_data_A, pool_size, pool_stride, padding)

no_filters_per_layer = [5,5,5]
filter_size = 5
stride = 1
output_data_A = cm.hidden_conv_layers_without_conv_mat(output_data_A,
                                                        stride, filter_size,
                                                        no_filters_per_layer,
                                                        'A',
                                                        first_layer_conv_mat = False)

output_data_A = pool_layer(pool_type, output_data_A, pool_size, pool_stride, padding)

###################################
# Type D crystal ###################

conv_D = np.load('conv_mat_D_with_middle.npy')
conv_D_rotated = np.load('conv_mat_D_rotated.npy') # for rotational symmetry

no_filters_per_layer = [10,10,10,10]
filter_size = 4
stride = 1
output_data_D = cm.hidden_conv_layers_without_conv_mat(x,
                                                        stride, filter_size,
                                                        no_filters_per_layer,
                                                       'D',
                                                       c_matrix= conv_D)


# Pooling of A (input, pooling size, strides, VALID means that it doesnt use padding)
output_data_D = pool_layer(pool_type, output_data_A, pool_size, pool_stride, padding)

no_filters_per_layer = [5,5,5]
filter_size = 5
stride = 1
output_data_D = cm.hidden_conv_layers_without_conv_mat(output_data_D,
                                                        stride, filter_size,
                                                        no_filters_per_layer,
                                                        'D',
                                                        first_layer_conv_mat = False)

output_data_D = pool_layer(pool_type, output_data_D, pool_size, pool_stride, padding)

# Concatenate output from convolution layers
y = tf.concat([output_data_A, output_data_D],2)

# Flatten the data to the fully-connected layers
y = tf.layers.flatten(y)

# Fully-connected layers
# Parameters
no_layers = 4
no_nodes_per_layer = [512, 256, 128, 64]
alpha = np.float32(0.001)
y = fcm.hidden_dense_layers(y, no_layers, no_nodes_per_layer, multiplicity, alpha, name = 'y')

lam_E = 1
lam_theta = 1
lam_phi = 1
E_offset = 0.1
beta = 0.0
use_relative_loss = 0
weight_decay = 0

# Define loss function
if lossfkn =='energy_theta_phi_permutation_loss':
    loss = loss_methods.energy_theta_phi_permutation_loss(y, y_, int(multiplicity/3), lam_E, lam_theta, lam_phi, E_offset, 0, 0, use_relative_loss, weight_decay, beta)
elif lossfkn == 'energy_theta_phi_permutation_loss_ricky':
    loss = loss_methods.energy_theta_phi_permutation_loss_ricky(y, y_, int(multiplicity/3))
elif lossfkn == "rickardtest":
    loss = loss_methods.loss_without_permutation(y,y_,int(multiplicity/3))
else:
    loss = loss_methods.energy_theta_phi_permutation_loss(y, y_, int(multiplicity/3))

# Define a training step
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Start session and initialize variables to train
print('Start training')
start_training_time = time.time()

sess.run(tf.global_variables_initializer())
#print("Memory currently used:", int(sess.run(bytes_in_use))/1e6,"MB")
saver = tf.train.Saver()

loss_value_eval = []
loss_value_train = []
for i in range(iterations):
    # Randomly selected events from the training set are extracted.
    x_batch, y_batch = cm.gen_sub_set(batch_size, det_data_train, gun_data_train)

    if i % 100 == 0: # don't want to save values at every training step

        #Evaluate on some random events
        x_batch_eval, y_batch_eval = cm.gen_sub_set(batch_size_eval, det_data_eval, gun_data_eval)

        # Evaluate the cost function value using the evaluation data
        loss_value = sess.run(loss, feed_dict={x: x_batch_eval, y_: y_batch_eval})

        # Evaluate the cost function on the training data.
        loss_value_train_tmp = sess.run(loss, feed_dict={x: x_batch, y_: y_batch})

        if i % 1000 == 0:
            print('Iteration nr. ', i, 'Loss: ', loss_value)
        if i % 10000 == 0:
            #print("Memory currently used:", int(sess.run(bytes_in_use))/1e6,"MB")
            pass
        loss_value_eval.append(loss_value)
        loss_value_train.append(loss_value_train_tmp)

    # Actual training step is done here
    sess.run(train_step, feed_dict={x: x_batch, y_: y_batch})
                                    #options=run_options,
                                    #run_metadata=run_metadata)


finish_training_time = time.time()
training_time = (finish_training_time-start_training_time)/60
print('Training done. Total runtime is:',training_time,'minutes.')
#print("Total memory used:", int(sess.run(bytes_in_use))/1e6,"MB")

# print starting time
print('training finished at:', datetime.datetime.now())

# Saving data and model
print("Saving model...")

list_names = ['loss_value_eval','loss_value_train']
ANN_save_load.save_results(saver, sess, message, list_names, loss_value_eval, loss_value_train,
                           save_folder = save_folder, training_time = training_time,
                           iterations = iterations, training_portion = training_portion,
                           max_multiplicity = int(multiplicity/3), used_data_portion = used_data_portion,
                           lam_E = lam_E, lam_theta = lam_theta, lam_phi = lam_phi,
                           E_offset = E_offset,
                           use_relative_loss = use_relative_loss)
print("Saving finished, closing session.")

"""
Memory tracing report
tf.profiler.profile(
    tf.get_default_graph(),
    options=tf.profiler.ProfileOptionBuilder.float_operation())
"""
sess.close()
# print starting time
print('saving plots finished at:', datetime.datetime.now())
