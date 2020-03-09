#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:48:37 2019

@author: Rickard Karlsson with contributions from Richard Martin, Jesper Jönsson and Martin Lidén

This script will train and evaluate a CNN and save the model + results to disk.

### How to use:
    specify input arguments for network structure as: "filters_per_layer_block_1,layers_block_1,filter_size_block_1,stride_block_1 filters_per_layer_block_2,layers_block_2,filter_size_block_2,stride_block_2 ... etc". For example to make a block with 10 filters per layer, 2 layers in the block, a filtersize of 5 elements and a stride of 3 we type: 10,2,5,3
    Then for each such block you want to have in the network you keep adding them like this and separate with a space, for example like this: 10,2,5,3 20,2,5,3 30,2,5,1

    You also need to specify the training data you want to use (npz_file_name parameter) and most likely some other stuff too. Scroll down to the line "#### Network parameters and arguments" to see a list of parameters you can tweak. Most of them is probably easiest to have hardcoded in the script when running tests but ofcourse this could be changed as needed.

    This script produces a folder containing the trained network model, a README.md file containing some info about the network and training process, a textfile named error.txt containing the mean errors and std devs from the reconstruction evaluation and some images of the resulting plots from the evaluation.



"""

""" #### Importing of modules and loading of functions and stuff ### """

# Installed python modules
import numpy as np
import tensorflow as tf
import time
import sys
import datetime
import importlib
import matplotlib.pyplot as plt
#from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse


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
cm = module_from_file("conv_methods", "../../CNN/Network/conv_methods.py")
ANN_save_load = module_from_file("ANN_save_load", "../../save_load/ANN_save_load.py")
fcm = module_from_file("FC_methods","../../FCN/FC_methods.py")
pm = module_from_file("plot_methods","../../plot_methods/plot_methods.py")
loss_methods=module_from_file("loss_functions","../../loss_functions/loss_functions.py")


# function that makes a pooling layer
def pool_layer(pool_type, output_data, pool_size, pool_stride=1, padding = 'VALID'):

    if pool_type == 'avg':
        return tf.nn.avg_pool(output_data, [1, 1, pool_size, 1], [1,1,pool_stride,1], padding = padding)
    elif pool_type == 'max':
        return tf.nn.max_pool(output_data, [1, 1, pool_size, 1], [1,1,pool_stride,1], padding = padding)
    else:
        return output_data

# function that builds a block of conv-layers and a pooling layer
def conv_layer(output_data_A, output_data_D):
    print('building',n,'th conv-block: filters per layer =',layer[0],', layers =',layer[1],', filter size =',filter_size,', stride =',stride)

    #conv type A crystal
    output_data_A = cm.hidden_conv_layers_without_conv_mat(output_data_A,
                                                    stride, filter_size,
                                                    no_filters_per_layer,
                                                    'A',
                                                    first_layer_conv_mat = False)

    #conv type D crystal
    output_data_D = cm.hidden_conv_layers_without_conv_mat(output_data_D,
                                                    stride, filter_size,
                                                    no_filters_per_layer,
                                                    'D',
                                                    first_layer_conv_mat = False)

    if pool_type != 'off':
        print('building pooling layer: pool type = ' +pool_type + ', pool size = ' +str(pool_size) + ', pool stride = ' +str(pool_stride) + ', padding = ' +padding)
    else:
        print('pooling is off')

    # pool of A
    output_data_A = pool_layer(pool_type, output_data_A, pool_size, pool_stride, padding)


    # pool of D
    output_data_D = pool_layer(pool_type, output_data_D, pool_size, pool_stride, padding)

    return output_data_A, output_data_D


# print starting time
print('imports loaded at:', datetime.datetime.now())

# GPU code (comment if you don't use GPU)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config=config)
#sess = tf.Session()

# Memory tracing
# with tf.device('/device:GPU:0'):  # Checks memory usage
#   bytes_in_use = BytesInUse()
#
# run_metadata = tf.RunMetadata()
# run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)


""" ####  Hardcoded parameters and arguments  ### """

    # training
iterations = int(3)         # the number of iterations to train for (ex 500000)
batch_size = 60             # How many events to train on in each iteration (ex 100)
batch_size_eval = 60        # How many events to evaluate on in each iteration (ex 100)
learning_rate = 0.0001      # optimizer parameter (should probably be left at 0.0001)
used_data_portion = 1       # How much of the training data that is used (should probably be 1)
training_portion = 0.999    # how much of the dataset to use for training, the rest is used for evaluation (should probably be around 0.8)
npz_file_name = '../../../training_data/XB_mixed_data_1-3_528339.npz' # The training dataset
message = str(sys.argv) + "\n" + 'Test av ...' # optional message that gets saved with the results

    # pooling (See our report for more info)
pool_size = 1
pool_stride = 1
padding = 'VALID'
pool_type = 'off'

    # loss function (See our report and the loss_functions folder for more info)
lossfkn = 'vanlig'
lam_E = 1
lam_theta = 1
lam_phi = 1
E_offset = 0.1
beta = 0
use_relative_loss = 0
weight_decay = 0

    # Fully connected structure (See our report for more info)
no_layers = 3
no_nodes_per_layer = [1024, 1024, 256]
alpha = np.float32(0.001)       #Slope for negative values for Leaky ReLu


""" ### Commandline parameters and arguments. ### """


# Convolutipnal network structure. Could also be hardcoded in this format: ('4,2,3,1', '8,2,3,1')
layer_data = []
for arg in sys.argv[1:]:
    layer_data.append(list(map(int, arg.split(','))))

# automatically names the savefolder like the network structure. Might not be optimal for long structures.
save_folder = ';'.join(sys.argv[1:])



""" ### Load data and build the network ### """

# print starting time
print('Starting building network at:', datetime.datetime.now())

# Get training and evaluation data
[det_data_train,
 gun_data_train,
 det_data_eval,
 gun_data_eval] = fcm.read_data_npz_energy_angle(npz_file_name,training_portion,
                                                    used_data_portion)

# Decide multiplicity from input data
multiplicity = np.shape(gun_data_eval)[1]

# print run-arguments to output
print(message)

# Create placeholder for input and correct values
x = tf.placeholder(dtype=tf.float32,shape=(None,cm.input_size), name = 'x')
y_ = tf.placeholder(dtype=tf.float32, shape=[None,multiplicity], name = 'y_')

# Convolutional layers
"""
Vill man använda conv_A_rotated eller conv_D_rotated prata
med Rickard om hur det implementeras
"""
# load convolution matrices
conv_A = np.load('../../CNN/Network/conv_mat_A.npy')
#conv_A_rotated = np.load('../../CNN/Network/conv_mat_A_rotated.npy') # for rotational symmetry
conv_D = np.load('../../CNN/Network/conv_mat_D.npy')
#conv_D_rotated = np.load('../../CNN/Network/conv_mat_D_rotated.npy') # for rotational symmetry


# create the convolution layers
first_layer = True
n=1
for layer in layer_data:
    # format layer lists and etract layer data
    no_filters_per_layer = [layer[0] for _ in range(layer[1])]
    filter_size = layer[2]
    stride = layer[3]

    # create the first layer with conv matrix
    if first_layer:
        print('building first conv-block: filter per layer =',layer[0],', layers =',layer[1],', filter size =',filter_size,', stride =',stride)

        #conv type A crystal
        output_data_A = cm.hidden_conv_layers_without_conv_mat(x,
                                                            stride, filter_size,
                                                            no_filters_per_layer,
                                                           'A',
                                                           c_matrix= conv_A)
        # conv type D crystal
        output_data_D = cm.hidden_conv_layers_without_conv_mat(x,
                                                            stride, filter_size,
                                                            no_filters_per_layer,
                                                            'D',
                                                            c_matrix = conv_D)

        if pool_type != 'off':
            print('building pooling layer: pool type = ' +pool_type + ', pool size = ' +str(pool_size) + ', pool stride = ' +str(pool_stride) + ', padding = ' +padding)
        else:
            print('pooling is off')

        # Pooling of A (input, pooling size, strides, VALID means that it doesnt use padding)
        output_data_A = pool_layer(pool_type, output_data_A, pool_size, pool_stride, padding)

        # Pooling of D
        output_data_D = pool_layer(pool_type, output_data_D, pool_size, pool_stride, padding)

        first_layer = False

    # create the rest of the layers without conv matrix
    else:
        n+=1
        output_data_A, output_data_D = conv_layer(output_data_A, output_data_D)


# Concatenate output from convolution layers
y = tf.concat([output_data_A, output_data_D],2)

# Flatten the data to the fully-connected layers
y = tf.layers.flatten(y)

# create Fully-connected layers
y = fcm.hidden_dense_layers(y, no_layers, no_nodes_per_layer, multiplicity, alpha, name = 'y')


# Define loss function
if lossfkn =='vanlig':
    loss = loss_methods.energy_theta_phi_permutation_loss(y, y_, int(multiplicity/3), lam_E, lam_theta, lam_phi, E_offset, 0, 0, use_relative_loss, weight_decay, beta)
elif lossfkn == 'ricky':
    loss = loss_methods.energy_theta_phi_permutation_loss_ricky(y, y_, int(multiplicity/3), lam_E, lam_theta, lam_phi, E_offset, 0, 0, use_relative_loss, weight_decay, beta)
elif lossfkn == "rickardtest":
    loss = loss_methods.loss_without_permutation(y,y_,int(multiplicity/3))
else:
    loss = loss_methods.energy_theta_phi_permutation_loss(y, y_, int(multiplicity/3), lam_E, lam_theta, lam_phi, E_offset, 0, 0, use_relative_loss, weight_decay, beta)

# Define a training step
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)


""" ### Initialize and start the training process ### """

# Start session and initialize variables to train
print('Structure built. Starting training at', datetime.datetime.now())
start_training_time = time.time()
sess.run(tf.global_variables_initializer())
#print("Memory currently used:", int(sess.run(bytes_in_use))/1e6,"MB")
saver = tf.train.Saver()
loss_value_eval = []
loss_value_train = []

# start the training loop
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
        if i % 1000 == 0:
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
print("Saving finished at:",datetime.datetime.now()," starting plotting.")


""" ### The network is now trained and saved. The rest of this script is evaluation and saving of evaluation results. ### """


# evaluation Parameters
no_bins = 30
max_energy_angle_values = [10, np.pi, 2*np.pi]
multiplicity = int(np.shape(gun_data_eval)[1]/3)
#n = 1 #len(folder_names)
#m = len(list_names)-1
#k = 0


# Reconstruction of the evaluation data set by using the trained network:
print("Running network with evaluation data")
gun_data_from_network = np.empty((0,9))
for i in range(0,det_data_eval.shape[0],100):
    output_from_network = np.asarray(sess.run(y, feed_dict={x: det_data_eval[i:i+100][:]}))
    gun_data_from_network = np.concatenate((gun_data_from_network, output_from_network), axis = 0)

gun_data_eval = gun_data_eval[0:gun_data_from_network.shape[0]]


# extract energy and angles input and network-output
events = loss_methods.network_permutation_sort(gun_data_from_network,gun_data_eval, lam_E, lam_theta, lam_phi, E_offset)
for key in events:
    values = np.asarray(events.get(key))
    events.update({key:values})
events.update({4: np.mod(events[4], 2*np.pi)})

# calculate error and performance measures
mean_error, error_each_bin, mean_std, std_each_bin = pm.error_measurement(events, no_bins, max_energy_angle_values, print_performance = True)
print("Reconstruction finished at:" ,datetime.datetime.now())

# save errors and performance plots
print("Saving images to folder:", save_folder)
full_path = './'+save_folder

# save error measurements
err_E = 'mean_E = ' + str(np.around(mean_error[0],3))
err_theta = 'mean_\u03F4 = ' + str(np.around(mean_error[1],3))
err_phi = 'mean_\u03A6 = ' + str(np.around(mean_error[2],3))
std_E = 'std_E = ' + str(np.around(mean_std[0],3))
std_theta = 'std_\u03F4 = ' + str(np.around(mean_std[1],3))
std_phi = 'std_\u03A6 = ' + str(np.around(mean_std[2],3))

error_file = open(full_path+'/error.txt', 'a+')
error_file.write('\n' + err_E + '\n' +  err_theta + '\n' +  err_phi + '\n' +  std_E + '\n' +  std_theta + '\n' +  std_phi + '\n')
error_file.close()

# save histograms
fig, axs = pm.energy_theta_phi_splitted(events[0], events[2], events[4], events[1], events[3], events[5],
                         name=save_folder, bins = 500, save=True)
fig.savefig(full_path + '/hist.png')
plt.close(fig)

# save errorplots
perffig, perfaxs = pm.error_plot(error_each_bin, std_each_bin, no_bins, max_energy_angle_values, name=save_folder, save=True)
perffig.savefig(full_path + '/errors.png')
plt.close(perffig)

# save loss
listfig, ax = plt.subplots()
ax.plot(loss_value_train)
ax.plot(loss_value_eval)
ax.set(ylabel = 'Train and evaluation loss', xlabel = 'Iteration (%)')
listfig.suptitle(save_folder)
plt.savefig(full_path + '/loss_lists.png')
plt.close(listfig)

"""
Memory tracing report
tf.profiler.profile(
    tf.get_default_graph(),
    options=tf.profiler.ProfileOptionBuilder.float_operation())
"""
sess.close()
# print ending time
print('Plot-saving finished at:', datetime.datetime.now())
