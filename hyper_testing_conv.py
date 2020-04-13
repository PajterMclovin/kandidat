""" @PETER HALLDESTAM, 2020
    
    Runs different hyperparameter tests

"""
import os
import json
import numpy as np
from datetime import datetime as dt


from plotting import plot_predictions
from plotting import plot_loss
from utils import load_data
from utils import get_eval_data
from utils import get_permutation_match
from utils import cartesian_to_spherical
from utils import get_measurement_of_performance
from utils import get_no_trainable_parameters
from utils import get_trained_model

GEANT4_DATA = ('home/david/', 'XB_mixed_data_1-2_653348.npz')
EVALUATION_SPLIT = .1
NO_EPOCHS = 1
DEPTHS = [2]

def depth_test():
    """
    Tests the momentum loss in FCNs with different depths. The measurement of 
    performance is the standard deviation in the prediction error evaluated for
    each evaluation set in EVALUATION_NPZS.

    """
    train_data, train_labels = {}, {}
    eval_data, eval_labels = {}, {}
    max_mult = []
    meas_of_perf = {}
    
    #create save directory
    save_dir = 'depth_test_' + dt.now().strftime("%d-%b-%Y(%H.%M.%S)")
    main_dir = os.path.join(os.getcwd(), save_dir)
    os.makedirs(main_dir)    
    
    #load training(validation) and evaluation sets with varying max multiplicity
    for i in range(len(GEANT4_DATA)):
        print('Loading no. {}/{} for training and evaluation.'.format(i+1, len(GEANT4_DATA)))
        data, labels = load_data(GEANT4_DATA[i], 1.0, cartesian=True)
        train_data[i], train_labels[i], eval_data[i], eval_labels[i] = get_eval_data(data, labels, EVALUATION_SPLIT)
        max_mult.append(int(len(labels[0])/3))
        
    # if len(max_mult) > len(set(max_mult)):
    #     raise ValueError('max_mult must be different in each data/label set!')
    
    #iterate through all max multiplicities
    for i in range(len(GEANT4_DATA)):
        m = max_mult[i]
        
        meas_of_perf[m] = {}
        max_mult_dir = os.path.join(main_dir, 'max_mult={}'.format(m) + dt.now().strftime("%d-%b-%Y(%H.%M.%S)"))
        os.makedirs(max_mult_dir)
        
        #iterate through all depths
        for j in range(len(DEPTHS)):
            
            print('Training network with depth {}, network no. {}/{}'.format(DEPTHS[j], j+1, len(DEPTHS)))
            trained_model, training = get_trained_model(train_data[i], train_labels[i], no_epochs=NO_EPOCHS, depth=DEPTHS[j])
            no_params = get_no_trainable_parameters(trained_model)
            print(trained_model.summary())
            print('Evaluating...')
            predictions = trained_model.predict(eval_data[i])
            y = cartesian_to_spherical(predictions, error=True)
            y_ = cartesian_to_spherical(eval_labels[i], error=True)
            y, y_ = get_permutation_match(y, y_, True)
            mean_std = get_measurement_of_performance(y, y_)
            meas_of_perf[m][DEPTHS[j]] = mean_std
            
            print('Saving plots etc...')
            learning_curve = plot_loss(training)
            prediction_plot, rec_events = plot_predictions(y, y_)
            title = """depth: {}, max_mult: {}, trainable parameters: {}, epochs: {},
                    mean: {}, standard deviation: {},
                    #events: {} (training) {} (evaluation, shown)
                    """.format(DEPTHS[j], m, no_params, NO_EPOCHS, mean_std['mean'], 
                               mean_std['std'], len(train_data), len(eval_data))
            learning_curve.suptitle(title)
            prediction_plot.suptitle(title)
            learning_curve.savefig(os.path.join(max_mult_dir, 'LC_depth={}'.format(DEPTHS[j])))
            prediction_plot.savefig(os.path.join(max_mult_dir, 'PP_depth={}'.format(DEPTHS[j])))

     
    #save dictionary containing the measurements of performance into json-file
    save_to = os.path.join(main_dir, 'meas_of_perf')
    with open(save_to, 'w') as save_file:
        json.dump(meas_of_perf, save_file, sort_keys=True, indent=4)
            
 
