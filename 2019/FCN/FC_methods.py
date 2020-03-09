import numpy as np
import tensorflow as tf
import itertools as it
import importlib, importlib.util
import scipy as sp
#####################################################
################# OTHER METHODS #####################
#####################################################

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

#####################################################
############## FULLY CONNECTED LAYERS ###############
#####################################################

#Constructs fully connected layers with {no_hidden_layers}+2 layers. Uses Leaky-ReLu.
    #x = input data
    #no_hidden_layers = the total number of layers -2 because of initial and final layer
    #no_nodes_per_layer = input shape either scalar number or array (eg. [182, 34, 35]) with length {no_hidden_layers}
    #no_output_nodes  = number of output nodes, for example 2*{maximum multiplicity of the batch}
    #alpha = the slope for the Leaky-ReLu, 0 < alpha < 1
def hidden_dense_layers(x_input, no_hidden_layers, no_nodes_per_layer, no_output_nodes, alpha, dropout_rate = 0.2, name = 'y'):
    W, b = {}, {}
    if len(no_nodes_per_layer) == 1:
        no_nodes_per_layer = np.ones(no_hidden_layers, dtype = np.int32)*no_nodes_per_layer
     
    #Initial layer
    W['W' + str(1)] = tf.Variable(tf.truncated_normal([int(x_input.shape[1]), no_nodes_per_layer[0]], stddev=0.1), dtype=tf.float32) ###
    b['b' + str(1)] = tf.Variable(tf.ones([no_nodes_per_layer[0]]), dtype=tf.float32)
    #Middle layers
    for i in range(1,no_hidden_layers):
        W['W'+str(i+1)]=tf.Variable(tf.truncated_normal([no_nodes_per_layer[i-1], no_nodes_per_layer[i]], stddev=0.1), dtype=tf.float32)
        b['b'+str(i+1)]=tf.Variable(tf.ones([no_nodes_per_layer[i]]), dtype=tf.float32)
    #Final layer
    W["W" + str(no_hidden_layers+1)] = tf.Variable(tf.truncated_normal([no_nodes_per_layer[no_hidden_layers-1], no_output_nodes], stddev=0.1),dtype=tf.float32)
    b["b" + str(no_hidden_layers+1)] = tf.Variable(tf.ones([no_output_nodes]), dtype=tf.float32)
    
    y = x_input
    for i in range(no_hidden_layers):
        y=tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(y, W["W"+str(i+1)]) + b["b"+str(i+1)], alpha), keep_prob = 1 - dropout_rate)         
    y=tf.matmul(y,W["W"+str(no_hidden_layers+1)]) + b["b"+str(no_hidden_layers+1)]
    y = tf.identity(y, name = name)
    
    no_nodes = str(x_input.shape[1]) + ' + ' + str(no_nodes_per_layer) + ' + ' + str(no_output_nodes)
    total_number_of_nodes = int(x_input.shape[1] + sum(no_nodes_per_layer) + no_output_nodes)
    print('Network structure: #Nodes = ' + no_nodes + ' = ' + str(total_number_of_nodes))
    print()
    return y

#Same as above but for weight decay; also returning weight regularization loss
def hidden_dense_layers_weight_decay(x_input, no_hidden_layers, no_nodes_per_layer, no_output_nodes, alpha, dropout_rate = 0.2, name = 'y'):
    W, b = {}, {}
    weight_decay_loss = 0
    
    if len(no_nodes_per_layer) == 1:
        no_nodes_per_layer = np.ones(no_hidden_layers, dtype = np.int32)*no_nodes_per_layer
     
    #Initial layer
    W['W' + str(1)] = tf.Variable(tf.truncated_normal([int(x_input.shape[1]), no_nodes_per_layer[0]], stddev=0.1), dtype=tf.float32) ###
    b['b' + str(1)] = tf.Variable(tf.ones([no_nodes_per_layer[0]]), dtype=tf.float32)
    #Middle layers
    for i in range(1,no_hidden_layers):
        W['W'+str(i+1)]=tf.Variable(tf.truncated_normal([no_nodes_per_layer[i-1], no_nodes_per_layer[i]], stddev=0.1), dtype=tf.float32)
        b['b'+str(i+1)]=tf.Variable(tf.ones([no_nodes_per_layer[i]]), dtype=tf.float32)
        weight_decay_loss = weight_decay_loss + tf.nn.l2_loss(W['W'+str(i+1)])
    #Final layer
    W["W" + str(no_hidden_layers+1)] = tf.Variable(tf.truncated_normal([no_nodes_per_layer[no_hidden_layers-1], no_output_nodes], stddev=0.1),dtype=tf.float32)
    b["b" + str(no_hidden_layers+1)] = tf.Variable(tf.ones([no_output_nodes]), dtype=tf.float32)
    
    y = x_input
    for i in range(no_hidden_layers):
        y=tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(y, W["W"+str(i+1)]) + b["b"+str(i+1)], alpha), keep_prob = 1 - dropout_rate) 
        #y=tf.nn.dropout(tf.nn.relu(tf.matmul(y, W["W"+str(i+1)]) + b["b"+str(i+1)]), keep_prob = 1 - dropout_rate)        
    y=tf.matmul(y,W["W"+str(no_hidden_layers+1)]) + b["b"+str(no_hidden_layers+1)]
    y = tf.identity(y, name = name)
    
    no_nodes = str(x_input.shape[1]) + ' + ' + str(no_nodes_per_layer) + ' + ' + str(no_output_nodes)
    total_number_of_nodes = int(x_input.shape[1] + sum(no_nodes_per_layer) + no_output_nodes)
    print('Network structure: #Nodes = ' + no_nodes + ' = ' + str(total_number_of_nodes))
    print()
    
    return y, weight_decay_loss

#####################################################
############## DATA PROCESSING METHODS ##############
#####################################################
    
#Reads the relevant data from a npz file for an energy/theta output network
def read_data_npz_energy_angle(npz_file_name, training_portion, used_data_portion):
    print('Reading data...')
    data_set = np.load(npz_file_name)
    det_data = data_set['detector_data']
    labels = data_set['energy_labels']
    
    no_used_events = int(len(labels)*used_data_portion)
    no_train = int(training_portion*no_used_events)
    print('Done! Using', no_train, 'training events and', no_used_events-no_train, 'evaluation events.')
    return det_data[:no_train], labels[:no_train], det_data[no_train:no_used_events], labels[no_train:no_used_events]

#Returns a random subset of size {no_data} from the two data batches x & y
def subset(batch_x, batch_y, no_data):
    ind = np.random.randint(0, len(batch_x), no_data)
    return batch_x[ind], batch_y[ind]

'''Applies the Doppler correction on the energies for both the reconstructed and correct energies
E_network = reconstructed energies
E_labels = correct label energies
theta_network = reconstructed thetas
theta_labels = correct label thetas
rel_beta = speed of the relativistic ion beam
'''
def lab_E_to_beam_E(E_network, E_labels, theta_network, theta_labels, rel_beta):  
    E_prim_network = ((1-rel_beta*np.cos(theta_network))/(np.sqrt(1 - rel_beta*rel_beta)))*E_network
    E_prim_labels = ((1-rel_beta*np.cos(theta_labels))/(np.sqrt(1 - rel_beta*rel_beta)))*E_labels  
    return E_prim_network, E_prim_labels

#Adds events without detected gamma-rays to both detector data and corresponding labels
def add_zero_multiplicity(det_data, labels):
    max_multiplicity = int(len(labels[0])/3)

    new_labels = np.zeros((int(len(labels)*(max_multiplicity+1)/max_multiplicity), len(labels[0])))
    new_det_data = np.zeros((int(len(det_data)*(max_multiplicity+1)/max_multiplicity), 162))
        
    for i in range(int(len(labels)/max_multiplicity)):
        new_labels[(max_multiplicity + 1)*i + 1:(max_multiplicity + 1)*i + max_multiplicity + 1,:] = labels[max_multiplicity*i:max_multiplicity*i + max_multiplicity,:]
        new_det_data[(max_multiplicity + 1)*i + 1:(max_multiplicity + 1)*i + max_multiplicity + 1,:] = det_data[max_multiplicity*i:max_multiplicity*i + max_multiplicity,:]
    
    return new_labels, new_det_data

#Removes a periodically occuring row, probably a specific multiplicity. Use shift to switch with rows to be removed
def remove_last_multiplicity(det_data, labels, shift = 0):
    max_multiplicity = int(len(labels[0])/3)
    labels[0:len(labels)-shift,:] = labels[shift:len(labels),:]
    det_data[0:len(labels)-shift,:] = det_data[shift:len(labels),:]
    
    new_labels = np.zeros((int(len(labels)*(max_multiplicity-1)/max_multiplicity), len(labels[0])))
    new_det_data = np.zeros((int(len(det_data)*(max_multiplicity-1)/max_multiplicity), 162))
    
    for i in range(int(len(labels)/max_multiplicity)):
        new_labels[(max_multiplicity - 1)*i:(max_multiplicity - 1)*i + max_multiplicity - 1,:] = labels[(max_multiplicity)*i:(max_multiplicity)*i + max_multiplicity - 1,:]
        new_det_data[(max_multiplicity - 1)*i:(max_multiplicity - 1)*i + max_multiplicity - 1,:] = det_data[(max_multiplicity)*i:(max_multiplicity)*i + max_multiplicity - 1,:]
    
    return new_labels, new_det_data

#Divides network output and correct labels (energies and angles gathered) into splitted quantities gathered in events
def split_energy_angle(gun_data_from_network, gun_data_eval):
    energy_data = []
    energy_eval = []
    theta_data = []
    theta_eval = []
    phi_data = []
    phi_eval = []
    
    events = {0: energy_data,
              1: energy_eval,
              2: theta_data,
              3: theta_eval,
              4: phi_data,
              5: phi_eval}
    
    for event in gun_data_from_network:
        i = 0
        for value in event:
            if i%3 == 0:
                events.get(0).append(value)
            elif i%3 == 1:
                events.get(2).append(value)
            elif i%3 == 2:
                events.get(4).append(value)
            i += 1
    for event in gun_data_eval:
        i = 0
        for value in event:
            if i%3 == 0:
                events.get(1).append(value)
            elif i%3 == 1:
                events.get(3).append(value)
            elif i%3 == 2:
                events.get(5).append(value)
            i += 1
    
    #Cast events to numpy-arrays
    for key in events:
        values = np.asarray(events.get(key))
        events.update({key:values})
    
    return events