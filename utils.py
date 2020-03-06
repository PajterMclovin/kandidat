import numpy as np
from tensorflow.keras import layers
# import tensorflow as tf
# import itertools as it
# import importlib, importlib.util
# import scipy as sp

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

def res_net_block(input_data, filters, conv_size):
    x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, input_data])
    x = layers.Activation('relu')(x)
    return x

def non_res_block(input_data, filters, conv_size):
    x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x