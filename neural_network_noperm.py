""" @PETER HALLDESTAM, 2020
    
    Construction and training of a neural network. Easy to make changes to any 
    parameters and most importantly to implements different structures.
"""

from tensorflow.keras.optimizers import Adam
from time import time
import h5py

from models import FCN
import loss_functions as lf
from utils import load_data, get_eval_data
from plotting import plot_predictions

## ----------------------------- PARAMETERS -----------------------------------
NAME = 'test_model-{}'.format(int(time()))

NPZ_DATAFILE = 'test.npz'                       #or import sys and use sys.argv[1]
TOTAL_PORTION = 1.0                             #portion of file data to be used, (0,1]
EVAL_PORTION = 0.2                              #portion of total data for final evalutation (0,1)

NO_EPOCHS = 1                                  #Number of times to go through training data
BATCH_SIZE = 300                                #The training batch size
LEARNING_RATE = 1e-4                            #Learning rate/step size
VALIDATION_SPLIT = 0.1                          #??

def main():
    #load simulation data. OBS. labels need to be ordered in decreasing energy!
    data, labels = load_data(NPZ_DATAFILE, TOTAL_PORTION)
    
    #detach subset for final evaluation. Other is used for training and validation
    train_data, train_labels, eval_data, eval_labels = get_eval_data(data, labels, EVAL_PORTION)
    
    
    ### ------------- BUILD, TRAIN & TEST THE NEURAL NETWORK ------------------
    
    no_inputs = len(train_data[0])                  #no. input nodes (162 for each detector)
    no_outputs = len(train_labels[0])               #no. output nodes (3*max multiplicity) 
    
    #initiate the network structure
    model = FCN(no_inputs, no_outputs, 10, 128)
    
    #compile the model, choose loss function
    loss_function = lf.absolute_loss
    opt = Adam(lr=LEARNING_RATE)
    model.compile(optimizer=opt, loss=loss_function, metrics=['accuracy'])
    
    
    #train the model
    training = model.fit(train_data, train_labels, 
                         epochs=NO_EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT)
    
    #plot predictions
    predictions = model.predict(eval_data)
    figure, axes, rec_events = plot_predictions(predictions, eval_labels, permutation=False)
    figure.show()
    
    return model, predictions, training
if __name__ == '__main__':
    model, predictions, training = main()
