""" @PETER HALLDESTAM, 2020
    
    Construction and training of a neural network. Easy to make changes to any 
    parameters and most importantly to implements different structures.

"""

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import sys
import os
import pickle

from models import CNN
from loss_functions import loss_function_wrapper
from utils import load_data
from utils import get_eval_data
from utils import get_permutation_match
from utils import cartesian_to_spherical

from contextlib import redirect_stdout


## ----------------------------- PARAMETERS -----------------------------------

#### C-C-F NETWORK ####

NPZ_DATAFILE = sys.argv[1]+'.npz'                      #or import sys and use sys.argv[1]
TOTAL_PORTION = 1                                #portion of file data to be used, (0,1]
EVAL_PORTION = 0.1                              #portion of total data for final evalutation (0,1)
VALIDATION_SPLIT = 0.1                          #portion of training data for epoch validation
CARTESIAN = True                                #train with cartesian coordinates instead of spherical
CLASSIFICATION = False                          #train with classification nodes

NO_EPOCHS = int(sys.argv[2])
                                               #Number of times to go through training data
BATCH_SIZE = 2**8                                #The training batch size
LEARNING_RATE = 1e-4                            #Learning rate/step size
PERMUTATION = True                              #set false if using an ordered data set
LOSS_FUNCTION = 'mse'                           #type of loss: {mse, modulo, cosine} (only mse for cartesian)
MAT_SORT = "CCT"                                #type of sorting used for the convolutional matrix
USE_ROTATIONS = True
USE_REFLECTIONS = True

if sys.argv[3] == "T":
   USE_BATCH_NORMALIZATION = True 
else:
   USE_BATCH_NORMALIZATION = False 

FILTERS = [int(sys.argv[4]), int(sys.argv[5])]                            #must consist of even numbers!
DEPTH = int(sys.argv[6])    
                
def main():
    #name folder for save-files
    folder = "~/pfs/"  
    for i in range(len(sys.argv)-1):
        i = i+1
        folder = folder + sys.argv[i]
        if i < len(sys.argv)-1:
            folder = folder + "_"
    #make folder
    try:
        os.makedirs(folder)
        print("Skapapt mapp: "+folder)
    except FileExistsError:
        print("Invalid folder!")
    
    folder = folder + "/"
    
    #load simulation data. OBS. labels need to be ordered in decreasing energy!
    data, labels = load_data(NPZ_DATAFILE, TOTAL_PORTION, 
                             cartesian=CARTESIAN,
                             classification=CLASSIFICATION)
    
    #detach subset for final evaluation. train_** is for both training and validation
    train_data, train_labels, eval_data, eval_labels = get_eval_data(data, labels,
                                                                     eval_portion=EVAL_PORTION)
    
    
    ### ------------- BUILD, TRAIN & TEST THE NEURAL NETWORK ------------------
    
    
    #no. inputs/outputs based on data set
    no_inputs = len(train_data[0])                  
    no_outputs = len(train_labels[0])               
    
    #initiate the network structure

    model = CNN(no_inputs, no_outputs, sort = MAT_SORT, filters = FILTERS,
                depth = DEPTH, 
                rotations = USE_ROTATIONS, reflections = USE_REFLECTIONS,
                batch_normalization = USE_BATCH_NORMALIZATION)
    
    #select loss function
    loss_function = loss_function_wrapper(no_outputs, 
                                          loss_type=LOSS_FUNCTION, 
                                          permutation=PERMUTATION,
                                          cartesian=CARTESIAN,
                                          classification=CLASSIFICATION)
    
    #select optimizer
    opt = Adam(lr=LEARNING_RATE)
   
    #compile the network
    model.compile(optimizer=opt, loss=loss_function, metrics=['accuracy'])
    
    es = EarlyStopping(monitor='val_loss', patience=3)
    mcp = ModelCheckpoint(filepath=folder+'checkpoint', monitor='val_loss')
    
    training = model.fit(train_data, train_labels, 
                         epochs=NO_EPOCHS, batch_size=BATCH_SIZE,
                         validation_split=VALIDATION_SPLIT,
                         callbacks=[es, mcp])

    
    #plot predictions on evaluation data
    predictions = model.predict(eval_data)

    
    if CARTESIAN:
        predictions = cartesian_to_spherical(predictions)
        eval_labels = cartesian_to_spherical(eval_labels)    
    if PERMUTATION:
        predictions, labels = get_permutation_match(predictions, eval_labels, CARTESIAN, loss_type=LOSS_FUNCTION)
    
    
    #save weights
    model.save_weights(folder+'weights.h5')
    
    #save summary
    with open(folder+'modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
    
    #save history
    with open(folder+'traininghistory', 'wb') as file_pi:
        pickle.dump(training.history, file_pi)
    
    return

if __name__ == '__main__':
    main()
