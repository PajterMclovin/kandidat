""" @PETER HALLDESTAM, 2020

    Tests different depths and maximum multiplicities for the fully connected
    network. For each (depth, max_mult)-configuration a learning curve and
    "lasersvärd"-plot is saved into corresponding directories. The mean 
    squared error of the predictions is also calculated and later collectively
    saved to a json-file.

"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime as dt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from utils.models import FCN

from loss_function.loss import LossFunction
from loss_function.loss_functions import absolute_error

from utils.data_preprocess import load_data
from utils.data_preprocess import get_eval_data

from utils.help_methods import get_permutation_match
from utils.help_methods import cartesian_to_spherical
from utils.help_methods import get_no_trainable_parameters

from utils.plot_methods import plot_predictions
from utils.plot_methods import plot_loss
from utils.plot_methods import plot_depth_mult


data = os.path.join(os.getcwd(), 'data', 'XB_mixed_data_1-2_653348.npz')
GEANT4_DATA = (data, data)

EVALUATION_SPLIT = .1
VALIDATION_SPLIT = .1
LEARNING_RATE = 1e-4
BATCH_SIZE = 2**8
NO_EPOCHS = 2

DEPTHS = [0,1,2]
WIDTH = 128
RLOSS = 'squared'
EARLY_STOPPING = 6

train_data, train_labels = {}, {}
eval_data, eval_labels = {}, {}
max_mult = []
mean_error = {}

#create save directory
save_dir = 'depth_width_test_' + dt.now().strftime("%d-%b-%Y(%H.%M.%S)")
main_dir = os.path.join(os.getcwd(), save_dir)
os.makedirs(main_dir)    

#load training(validation) and evaluation sets with varying max multiplicity
for i, g_data in enumerate(GEANT4_DATA):
    print('Loading no. {}/{} for training and evaluation.'.format(i+1, len(GEANT4_DATA)))
    data, labels = load_data(g_data, 1.0, cartesian=True)
    train_data[i], train_labels[i], eval_data[i], eval_labels[i] = get_eval_data(data, labels, EVALUATION_SPLIT)
    max_mult.append(int(len(labels[0])/3))
    
no_inputs = len(data[0])                  
no_outputs = len(labels[0])

#iterate through all max multiplicities
for i, m in enumerate(max_mult):
    
    
    mean_error[m] = {}
    max_mult_dir = os.path.join(main_dir, 'max_mult={}_'.format(m) + dt.now().strftime("%d-%b-%Y(%H.%M.%S)"))
    os.makedirs(max_mult_dir)
    
    #iterate through all depths
    for depth in DEPTHS:
        
        print('Training network with max_mult {} and depth {}'.format(m, depth))
        model = FCN(no_inputs, no_outputs, depth, WIDTH)
        loss_function = LossFunction(m)
        model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=loss_function.get())
        early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING)
        training = model.fit(train_data[i], train_labels[i],
                             epochs=NO_EPOCHS, batch_size=BATCH_SIZE,
                             validation_split=VALIDATION_SPLIT,
                             callbacks=[early_stopping])
        learning_curve = plot_loss(training)
        
        print('Evaluating and saving plots...')            
        y = model.predict(eval_data[i])
        y, y_ = get_permutation_match(y, eval_labels[i], loss_function, m)
        px, px_ = y[::,0::3], y_[::,0::3]
        py, py_ = y[::,1::3], y_[::,1::3]
        pz, pz_ = y[::,2::3], y_[::,2::3]
        mean_error[m][depth] = np.mean(absolute_error(px, py, pz, px_, py_, pz_))
        
        y = cartesian_to_spherical(y, error=True)
        y_ = cartesian_to_spherical(y_, error=True)       
        prediction_plot, rec_events = plot_predictions(y, y_)
        

        no_params = get_no_trainable_parameters(model)
        epoch = early_stopping.stopped_epoch
        if epoch==0:
            epoch = NO_EPOCHS
        title = """max_mult: {}, depth: {}, trainable parameters: {}, epochs: {},
                mean error: {},
                #events: {} (training) {} (evaluation)
                """.format(m, depth, no_params, epoch, mean_error[m][depth],
                            len(train_data[i]), len(eval_data[i]))

        learning_curve.suptitle(title)
        prediction_plot.suptitle(title)
        learning_curve.savefig(os.path.join(max_mult_dir, 'LC_depth={}'.format(depth)))
        prediction_plot.savefig(os.path.join(max_mult_dir, 'PP_depth={}'.format(depth)))
        plt.close('all')
 
#save dictionary containing the measurements of performance into json-file
save_to = os.path.join(main_dir, 'mean_error')
with open(save_to, 'w') as save_file:
    json.dump(mean_error, save_file, sort_keys=True, indent=4)
mean_error = plot_depth_mult(save_to)
mean_error.savefig(save_to)

