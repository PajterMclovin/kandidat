""" @PETER HALLDESTAM, 2020

    Tests different depths and widths for the fully connected
    network. For each (depth, width)-configuration a learning curve and
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
from utils.plot_methods import plot_depth_width


data = os.path.join(os.getcwd(), 'data', 'XB_mixed_data_1-2_653348.npz')
GEANT4_DATA = data

EVALUATION_SPLIT = .1
VALIDATION_SPLIT = .1
LEARNING_RATE = 1e-4
BATCH_SIZE = 2**8
NO_EPOCHS = 5

DEPTHS = [0, 1, 2, 8, 10]
WIDTHS = [10, 60, 128]
RLOSS = 'squared'
EARLY_STOPPING = 6

train_data, train_labels = {}, {}
eval_data, eval_labels = {}, {}

mean_error = {}

#create save directory
save_dir = 'depth_test_' + dt.now().strftime("%d-%b-%Y(%H.%M.%S)")
main_dir = os.path.join(os.getcwd(), save_dir)
os.makedirs(main_dir)    

print('Loading data/label set for training and evaluation.')
data, labels = load_data(GEANT4_DATA, 1.0, cartesian=True)
train_data, train_labels, eval_data, eval_labels = get_eval_data(data, labels, EVALUATION_SPLIT)

max_mult = int(len(labels[0])/3)    
no_inputs = len(data[0])                  
no_outputs = len(labels[0])

for width in WIDTHS:
    
    mean_error[width] = {}
    
    width_dir = os.path.join(main_dir, 'width={}_'.format(width) + dt.now().strftime("%d-%b-%Y(%H.%M.%S)"))
    os.makedirs(width_dir)

    for depth in DEPTHS:
 
        print('Training network with width {} and depth {}.'.format(width, depth))
        model = FCN(no_inputs, no_outputs, depth, width)
        loss_function = LossFunction(max_mult)
        model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=loss_function.get())
        early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING)
        training = model.fit(train_data, train_labels,
                             epochs=NO_EPOCHS, batch_size=BATCH_SIZE, 
                             validation_split=VALIDATION_SPLIT,
                             callbacks=[early_stopping])
        learning_curve = plot_loss(training)
        
        print('Evaluating and saving plots...')            
        y = model.predict(eval_data)
        y, y_ = get_permutation_match(y, eval_labels, loss_function, max_mult)
        px, px_ = y[::,0::3], y_[::,0::3]
        py, py_ = y[::,1::3], y_[::,1::3]
        pz, pz_ = y[::,2::3], y_[::,2::3]
        mean_error[width][depth] = np.mean(absolute_error(px, py, pz, px_, py_, pz_))
        
        y = cartesian_to_spherical(y, error=True)
        y_ = cartesian_to_spherical(y_, error=True)       
        prediction_plot, rec_events = plot_predictions(y, y_)
        
        no_params = get_no_trainable_parameters(model)
        epoch = early_stopping.stopped_epoch
        if epoch==0:
            epoch = NO_EPOCHS
        title = """width: {}, depth: {}, trainable parameters: {}, epochs: {},
                mean error: {},
                #events: {} (training) {} (evaluation)
                """.format(width, depth, no_params, epoch, mean_error[width][depth],
                            len(train_data), len(eval_data))

        learning_curve.suptitle(title)
        prediction_plot.suptitle(title)
        learning_curve.savefig(os.path.join(width_dir, 'LC_depth={}'.format(depth)))
        prediction_plot.savefig(os.path.join(width_dir, 'PP_depth={}'.format(depth)))
        plt.close('all')
 
#save dictionary containing the measurements of performance into json-file
save_to = os.path.join(main_dir, 'mean_error')
with open(save_to, 'w') as save_file:
    json.dump(mean_error, save_file, sort_keys=True, indent=4)
mean_error = plot_depth_width(save_to)
mean_error.savefig(save_to)

