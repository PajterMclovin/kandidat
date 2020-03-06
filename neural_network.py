import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import datetime as dt
import sys

#import custom shit
from loss_functions import relative_loss
from utils import read_data_npz_energy_angle, subset


#IMPORTANT PARAMETERS (from last year)
#npz_file_name = sys.argv[1]             #String with name of the generated data
npz_file_name = 'XB_mixed_data_1-2_653348.npz'
no_layers = 10                          #Number of hidden layers (not including input and output layers)
no_nodes_per_layer = [128]              #Number of nodes per hidden layer (eg. [128] or [128,256,512])
no_iter = 10000                         #Number of training iterations
lam_E = 1                               #Weighting factors in cost function
lam_theta = 1
lam_phi = 1
E_offset = 0                            #Offset value in relative cost function
theta_offset = 0
phi_offset = 0
dropout_rate = 0                        #Rate of dropout, [0,1] (0 = dropout off)
used_data_portion = 1.0                 #The used amount data, [0,1]                                
use_relative_loss = 0                   #0 = NOT USE relative loss function, 1 = USE -||-                    
save_folder_name = 'TEST_NAME'          #Name of the directory with all the saved data

#OTHER PARAMETERS
training_portion = 0.8                  #The portion of the data used for training, [0,1]
alpha = 0.001                           #Slope for negative values for Leaky ReLu
batch_size = 300                        #The training batch size
loss_size = 300                         #Batch size for calculating loss during training
no_bins = 50                            #Number of bins to calculate standard deviation and error
beta = 0                                #Weighting factor for the weight decay loss (do not use = 0)
learning_rate = 1e-4                    #The learning rate
rel_beta = 0.7                          #The speed of the relativistic beam (only used for relativistic corrections)
message = 'Test_explanation'            #Text explaning the specific test in saved textfile

'''Loading data from the npz file (may not be working when not using phi?)
    train_data = detector data used for training [?x162]
    train_labels = label values corresponding to train_data
    eval_data = detector data used for evaluation [?x162]
    eval_labels = label values corresponding to eval_data'''
train_data, train_labels, eval_data, eval_labels = read_data_npz_energy_angle(npz_file_name, training_portion, used_data_portion)

no_output = len(train_labels[0])                    #Width of output layer (max multiplicity = no_output/3 when using phi)
no_input_nodes = len(train_data[0])                 #Width of input layer (162)
#max_E = np.round(np.amax(eval_labels),1)            #Maximum correct energy
#max_energy_angle_values = [max_E, np.pi, 2*np.pi]

weight_decay_loss = 1
#create custom loss function for Keras module

def custom_loss(num_splits, param):
    def loss2(y, y_):
        return relative_loss(y, y_, num_splits, param)
    return loss2


# NETWORK DESIGN
inputs = keras.Input(shape=(no_input_nodes,))
x = layers.Dense(162, activation='softmax')(inputs)
outputs = layers.Dense(no_output, activation='softmax')(x)

model = keras.Model(inputs, outputs)

#something to keep track of training...? Can easily be saved to keep track of all tests
# callbacks = [
#     # Write TensorBoard logs to `./logs` directory
#     keras.callbacks.TensorBoard(log_dir='./log/{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), write_images=True)
# ]

param = (lam_E, lam_theta, lam_phi,
         E_offset, theta_offset, phi_offset,
         weight_decay_loss, beta)
loss_func = custom_loss(int(no_output/3), param)


model.compile(optimizer=keras.optimizers.Adam(),
              loss=loss_func,
              metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=2)
