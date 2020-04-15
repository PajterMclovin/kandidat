""" @PETER HALLDESTAM, 2020
    
    Construction and training of a neural network. Easy to make changes to any 
    parameters and most importantly to implements different structures.

"""
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from utils.models import FCN
from utils.models import GCN
from utils.models import ResNet

from loss_function.loss import LossFunction

from utils.plot_methods import plot_predictions
from utils.plot_methods import plot_loss

from utils.data_preprocess import load_data
from utils.data_preprocess import get_eval_data

from utils.help_methods import save
from utils.help_methods import get_permutation_match
from utils.help_methods import cartesian_to_spherical
from utils.help_methods import get_no_trainable_parameters


      
        

## ----------------------------- PARAMETERS -----------------------------------

SAVE_NAME = 'moment_cartesian'

NPZ_DATAFILE = os.path.join(os.getcwd(), 'data', 'XB_mixed_data_1-2_653348.npz')   #or import sys and use sys.argv[1]
TOTAL_PORTION = .3                            #portion of file data to be used, (0,1]
EVAL_PORTION = 0.2                              #portion of total data for final evalutation (0,1)
VALIDATION_SPLIT = 0.1                          #portion of training data for epoch validation
CARTESIAN = True                                #train with cartesian coordinates instead of spherical
CLASSIFICATION = False                          #train with classification nodes

NO_EPOCHS = 100
                                   #Number of times to go through training data
BATCH_SIZE = 2**8                                #The training batch size
LEARNING_RATE = 1e-4                            #Learning rate/step size
PERMUTATION = True                              #set false if using an ordered data set

RLOSS = 'squared'
CLOSS = 'cross_entropy'


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
# model = FCN(no_inputs, no_outputs, 10, 128,
#             cartesian=CARTESIAN,
#             classification=CLASSIFICATION)


model = FCN(no_inputs, no_outputs, 2, 6)


#select loss function
max_mult = 2
loss_function = LossFunction(max_mult)

#select optimizer
opt = Adam(lr=LEARNING_RATE)

#compile the network
model.compile(optimizer=opt, loss=loss_function.get(), metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3)


#train the network with training data
training = model.fit(train_data, train_labels, 
                     epochs=NO_EPOCHS, batch_size=BATCH_SIZE,
                     validation_split=VALIDATION_SPLIT,
                     callbacks=[early_stopping])

#plot the learning curve
learning_curve = plot_loss(training)
 

#plot predictions on evaluation data
predictions = model.predict(eval_data)

 

if PERMUTATION:
    predictions, eval_labels = get_permutation_match(predictions, eval_labels, loss_function, max_mult)

predictions = cartesian_to_spherical(predictions, error=True)
eval_labels = cartesian_to_spherical(eval_labels, error=True)       

#plot the "lasersvärd"
figure, rec_events = plot_predictions(predictions, eval_labels, 
                                            show_detector_angles=True)

# #add title
# no_params = get_no_trainable_parameters(model)
# title = """trainable parameters: {}, epochs: {}, loss: {}, 
#            cartesian: {}, permutation: {}, max_mult: {},
#            #events: {} (training) {} (evaluation, shown)
#         """.format(no_params, NO_EPOCHS, LOSS_FUNCTION, 
#                    CARTESIAN, PERMUTATION, int(no_outputs/3),
#                    len(train_data), len(eval_data))
# figure.suptitle(title)

#save figures and trained parameters
# save(SAVE_NAME, figure, learning_curve, model)



