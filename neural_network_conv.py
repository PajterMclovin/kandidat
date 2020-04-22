""" @PETER HALLDESTAM, 2020
    
    Construction and training of a neural network. Easy to make changes to any 
    parameters and most importantly to implements different structures.

"""

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import sys

from models import CNN
from loss_functions import loss_function_wrapper
from plotting import plot_predictions
from plotting import plot_loss
from utils import load_data
from utils import save
from utils import get_eval_data
from utils import get_permutation_match
from utils import cartesian_to_spherical
from utils import get_no_trainable_parameters

## ----------------------------- PARAMETERS -----------------------------------

NPZ_DATAFILE = 'test.npz'                        #or import sys and use sys.argv[1]
TOTAL_PORTION = 1                                #portion of file data to be used, (0,1]
EVAL_PORTION = 0.1                              #portion of total data for final evalutation (0,1)
VALIDATION_SPLIT = 0.1                          #portion of training data for epoch validation
CARTESIAN = True                                #train with cartesian coordinates instead of spherical
CLASSIFICATION = False                          #train with classification nodes

NO_EPOCHS = 200
                                               #Number of times to go through training data
BATCH_SIZE = 2**8                                #The training batch size
LEARNING_RATE = 1e-4                            #Learning rate/step size
PERMUTATION = True                              #set false if using an ordered data set
LOSS_FUNCTION = 'mse'                           #type of loss: {mse, modulo, cosine} (only mse for cartesian)
MAT_SORT = "CCT"                                #type of sorting used for the convolutional matrix
USE_ROTATIONS = True
USE_REFLECTIONS = True
FILTERS = [256, 16, 4]                          #must consist of even numbers!
def main():
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
                rotations = USE_ROTATIONS, reflections = USE_REFLECTIONS)
    
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
    model.summary()
    
    callback = EarlyStopping(monitor='val_loss', patience=3)
    
    training = model.fit(train_data, train_labels, 
                         epochs=NO_EPOCHS, batch_size=BATCH_SIZE,
                         validation_split=VALIDATION_SPLIT)
    
    #plot the learning curve
    learning_curve = plot_loss(training)
     
    
    #plot predictions on evaluation data
    predictions = model.predict(eval_data)

    
    if CARTESIAN:
        predictions = cartesian_to_spherical(predictions)
        eval_labels = cartesian_to_spherical(eval_labels)    
    if PERMUTATION:
        predictions, labels = get_permutation_match(predictions, eval_labels, CARTESIAN, loss_type=LOSS_FUNCTION)

    
    
    #plot the "lasersvärd"
    figure, rec_events = plot_predictions(predictions, eval_labels, 
                                                show_detector_angles=True)
    
    #add title
    no_params = get_no_trainable_parameters(model)
    title = """trainable parameters: {}, epochs: {}, loss: {}, 
               cartesian: {}, permutation: {}, max_mult: {},
               #events: {} (training) {} (evaluation, shown)
            """.format(no_params, NO_EPOCHS, LOSS_FUNCTION, 
                       CARTESIAN, PERMUTATION, int(no_outputs/3),
                       len(train_data), len(eval_data))
    figure.suptitle(title)
    
    save('/home/david/', figure, learning_curve, model)
    
    return

if __name__ == '__main__':
    main()
