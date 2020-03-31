""" @PETER HALLDESTAM, 2020
    
    Construction and training of a neural network. Easy to make changes to any 
    parameters and most importantly to implements different structures.

"""

from tensorflow.keras.optimizers import Adam
from time import time
from sys import argv
import h5py

from models import FCN
from loss_functions import loss_function_wrapper
from utils import load_data, get_eval_data
from plotting import plot_predictions

## ----------------------------- PARAMETERS -----------------------------------
NAME = 'test_model-{}'.format(int(time()))

NPZ_DATAFILE = 'test.npz'
TOTAL_PORTION = 1.0                             #portion of file data to be used, (0,1]
EVAL_PORTION = 0.1                              #portion of total data for final evalutation (0,1)
VALIDATION_SPLIT = 0.1                          #portion of training data for epoch validation
CARTESIAN = True                                #train with cartesian coordinates instead of spherical

NO_EPOCHS = 1                                   #Number of times to go through training data
BATCH_SIZE = 300                                #The training batch size
LEARNING_RATE = 1e-4                            #Learning rate/step size
PERMUTATION = True                              #set false if using an ordered data set
LOSS_FUNCTION = 'mse'                           #type of loss: {mse, modulo, vector, cosine}

## FCN params ##
DEPTH = 10
WIDTH = 128

def main():
    #load simulation data. OBS. labels need to be ordered in decreasing energy!
    data, labels = load_data(NPZ_DATAFILE, TOTAL_PORTION, cartesian_coordinates=CARTESIAN)
    
    #detach subset for final evaluation. train_** is for both training and validation
    train_data, train_labels, eval_data, eval_labels = get_eval_data(data, labels,
                                                                     eval_portion=EVAL_PORTION)
    
    
    ### ------------- BUILD, TRAIN & TEST THE NEURAL NETWORK ------------------
    
    no_inputs = len(train_data[0])                  #no. input nodes (162 for each detector)
    no_outputs = len(train_labels[0])               #no. output nodes (3*max multiplicity) 
    
    #initiate the network structure
    model = FCN(no_inputs, no_outputs, DEPTH, WIDTH)
    
    #select loss function
    loss_function = loss_function_wrapper(int(no_outputs/3), 
                                          loss_type=LOSS_FUNCTION, 
                                          permutation=PERMUTATION,
                                          cartesian_coordinates=CARTESIAN)
    
    #select optimizer
    opt = Adam(lr=LEARNING_RATE)
    
    #compile the network
    model.compile(optimizer=opt, loss=loss_function, metrics=['accuracy'])
    
    #train the network with training data
    training = model.fit(train_data, train_labels, 
                         epochs=NO_EPOCHS, batch_size=BATCH_SIZE,
                         validation_split=VALIDATION_SPLIT)
    
    #plot predictions on evaluation data
    predictions = model.predict(eval_data)
    figure, axes, rec_events = plot_predictions(predictions, eval_labels, 
                                                permutation=PERMUTATION,
                                                cartesian_coordinates=CARTESIAN,
                                                loss_type=LOSS_FUNCTION)
    
    figure.save_fig(NAME)   #does this work?
    return model, predictions, training

if __name__ == '__main__':
    model, predictions, training = main()
