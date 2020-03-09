""" @PETER HALLDESTAM, 2020
    
    Construction and training of a neural network. Easy to make changes to any parameters and most importantly:
    the network structure.

"""
from tensorflow import keras
from tensorflow.keras import layers
# import numpy as np
# import datetime as dt
# import sys

#import custom stuff
from loss_functions import Relative_loss        # custom keras.losses.Loss-object
from utils import read_data_npz_energy_angle    # to load data


#IMPORTANT PARAMETERS (stolen from last years rep)
#npz_file_name = sys.argv[1]             #String with name of the generated data
npz_file_name = 'XB_mixed_data_1-2_653348.npz'
no_iter = 10000                         #Number of training iterations
lam_E = 1                               #Weighting factors in cost function
lam_theta = 1
lam_phi = 1
E_offset = 0                            #Offset value in relative cost function
theta_offset = 0
phi_offset = 0
dropout_rate = 0                        #Rate of dropout, [0,1] (0 = dropout off)
used_data_portion = 1.0                 #The used amount data, [0,1]                                                   
save_folder_name = 'TEST_NAME'          #Name of the directory with all the saved data

training_portion = 0.8                  #The portion of the data used for training, [0,1]
alpha = 0.001                           #Slope for negative values for Leaky ReLu
no_epochs = 1                           #Number of times to go through training data
batch_size = 300                        #The training batch size
beta = 0                                #Weighting factor for the weight decay loss (do not use = 0)

'''
Loading data from the npz file (may not be working when not using phi?)
train_data = detector data used for training [?x162]
train_labels = label values corresponding to train_data
eval_data = detector data used for evaluation [?x162]
eval_labels = label values corresponding to eval_data
'''
train_data, train_labels, eval_data, eval_labels = read_data_npz_energy_angle(npz_file_name, training_portion, used_data_portion)

no_output = len(train_labels[0])                    #Width of output layer (max multiplicity = no_output/3 when using phi)
no_input_nodes = len(train_data[0])                 #Width of input layer (162)
#max_E = np.round(np.amax(eval_labels),1)            #Maximum correct energy
#max_energy_angle_values = [max_E, np.pi, 2*np.pi]

weight_decay_loss = 1   #what's this for?




# -----------------NETWORK DESIGN-----------------------

## define layers with Layer-objects (keras)
inputs = keras.Input(shape=(no_input_nodes,))
x = layers.Dense(128, activation='softmax')(inputs)
for i in range(9):
    x = layers.Dense(128, activation='softmax')(x)
outputs = layers.Dense(no_output, activation='softmax')(x)

## create the network
model = keras.Model(inputs, outputs)

## select loss function with Loss-object (keras)
num_splits = int(no_output/3)
loss_func = Relative_loss(num_splits,
                          lam_E, lam_theta, lam_phi,
                          E_offset, theta_offset, phi_offset,
                          weight_decay_loss, beta)

## compile model
model.compile(optimizer=keras.optimizers.Adam(),
              loss=loss_func,
              metrics=['accuracy'])

## train the network
model.fit(train_data, train_labels, epochs=10, batch_size=batch_size)

## evaluate, save etc.
