""" @PETER HALLDESTAM, 2020
    
    Construction and training of a neural network. Easy to make changes to any 
    parameters and most importantly to implements different structures.

"""

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import sys
import os
import pickle
import time
start = time.time()

from models import CN_FCN
from loss_functions import loss_function_wrapper
from utils import load_data
from utils import get_eval_data
from utils import get_permutation_match
from utils import cartesian_to_spherical
from utils import get_measurement_of_performance
from contextlib import redirect_stdout


## ----------------------------- PARAMETERS -----------------------------------

#### C-F-C-F NETWORK ####

NPZ_DATAFILE = 'Data/'+sys.argv[1]+'.npz'                #or import sys and use sys.argv[1]
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
DEPTH = [int(sys.argv[6]), int(sys.argv[7])]   
                
def main():
    folder = "/"  
    for i in range(len(sys.argv)-1):
        i = i+1
        folder = folder + sys.argv[i]
        if i < len(sys.argv)-1:
            folder = folder + "_"
    #make folder
    subf=0
    folder_created = False
    while folder_created == False:
        try:
            if subf == 0:
                string = ""
            else:
                string = "/"+str(subf)
            os.makedirs(os.getcwd()+"/Resultat"+folder+string)
            folder_created = True
        except FileExistsError:
            subf += 1
        if subf>20:
            print("Fixa dina mappar!")
    folder = os.getcwd()+"/Resultat"+folder+string
    print("Skapat mapp: ", folder)
    
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

    model = CN_FCN(no_inputs, no_outputs, sort = MAT_SORT, filters = FILTERS,
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
    
    es = EarlyStopping(monitor='val_loss', patience=5)
    mcp = ModelCheckpoint(filepath=folder+'/checkpoint', monitor='val_loss')
    
    training = model.fit(train_data, train_labels, 
                         epochs=NO_EPOCHS, batch_size=BATCH_SIZE,
                         validation_split=VALIDATION_SPLIT,
                         callbacks=[es, mcp])

    epochs = es.stopped_epoch
    if epochs == 0:
        epocs = NO_EPOCHS
    #plot predictions on evaluation data
    predictions = model.predict(eval_data)

    if CARTESIAN:
        predictions = cartesian_to_spherical(predictions)
        eval_labels = cartesian_to_spherical(eval_labels)    
    if PERMUTATION:
        predictions, labels = get_permutation_match(predictions, eval_labels, CARTESIAN, loss_type=LOSS_FUNCTION)
    
    #save weights
    model.save_weights(folder+'/weights.h5')
    
    #save summary and time
    with open(folder+'/modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            end = time.time()
            hours, rem = divmod(end-start, 3600)
            minutes, seconds = divmod(rem, 60)
            print("Elapsed time: ")
            print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
            print("Elapsed epochs: ", epochs)
            model.summary()
    
    #save history
    with open(folder+'/traininghistory', 'wb') as file_pi:
        pickle.dump(training.history, file_pi)
        
    #save predicted events and measurement of performance
    y = predictions
    y_ = eval_labels
    events = {'predicted_energy': y[::,0::3].flatten(),
              'correct_energy': y_[::,0::3].flatten(), 
              
              'predicted_theta': y[::,1::3].flatten(),
              'correct_theta': y_[::,1::3].flatten(),
              
              'predicted_phi': np.mod(y[::,2::3], 2*np.pi).flatten(),
              'correct_phi': y_[::,2::3].flatten()}
    np.save(folder+'/events',events)
    mop = get_measurement_of_performance(y, y_)
    np.save(folder+'/mop',mop)
    return

if __name__ == '__main__':
    main()
