""" @PETER HALLDESTAM, 2020
    
    Construction and training of a neural network. Easy to make changes to any 
    parameters and most importantly to implements different structures.

"""

from tensorflow.keras.callbacks import TensorBoard #,CSVlogger
from tensorflow.keras.optimizers import Adam
from time import time
import matplotlib.pyplot

from models import FCN
from loss_functions import relative_loss
from utils import load_data, plot_predictions

## ----------------------------- PARAMETERS -----------------------------------
NAME = 'test_model-{}'.format(int(time()))

NPZ_DATAFILE = 'test.npz'                       #or import sys and use sys.argv[1]
TOTAL_PORTION = 1.0                             #portion of total data to be used, [0,1]
TRAIN_PORTION = 0.8                             #portion of used data for training, [0,1]

NO_EPOCHS = 5                                   #Number of times to go through training data
BATCH_SIZE = 300                                #The training batch size


def main():
    #load simulation data. OBS. labels need to be ordered in decreasing energy!
    train_data, train_labels, eval_data, eval_labels = load_data(NPZ_DATAFILE, 
                                                                 TOTAL_PORTION,
                                                                 TRAIN_PORTION)
    
    
    
    ### ------------- BUILD, TRAIN & TEST THE NEURAL NETWORK ------------------
    
    no_inputs = len(train_data[0])                  #no. input nodes (162 for each detector)
    no_outputs = len(train_labels[0])               #no. output nodes (3*max multiplicity) 
    
    #initiate the network structure
    model = FCN(no_inputs, no_outputs, 10, 128)
    
    #compile the model, choose loss function
    model.compile(optimizer=Adam(), loss=relative_loss, metrics=['accuracy'])
    
    
    #for analyzing the learning curve etc.
    print('tensorboard saved as: ' + NAME)
    tensorboard = TensorBoard(log_dir='logs\{}'.format(NAME))
    
    #train the model
    model.fit(train_data, train_labels, epochs=NO_EPOCHS, batch_size=BATCH_SIZE, callbacks=[tensorboard])
    
    #plot predictions
    predictions = model.predict(eval_data)
    figure, axes, rec_events = plot_predictions(predictions, eval_labels)
    figure.show()

if __name__ == '__main__':
    main()
