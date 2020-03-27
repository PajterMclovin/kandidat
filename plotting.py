""" @PETER HALLDESTAM, 2020
    
    Plot methods to analyze the neural network
    
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from utils import get_permutation_match


def plot_predictions(y, y_, bins=500, permutation=True):
    """
    Use to plot a models predictions in similar format as previous years, i.e. 2d histograms ("lasersvärd")
    
    Args:
        prediction : use model.predict(data)
        labels : to compare with predictions
        bins : number of bins i histogram
        permutation : True if network used permutational loss, False if not
    Returns:
        figure, axes, events (dict)
    Raises : if data and labels is not the same length
        ValueError : if prediction and labels is not of same length
    """
    if not len(y)==len(y_):
        raise TypeError('The prediction must be of same length as labels.') 
      
    if permutation:
        y, y_ = get_permutation_match(y, y_)
        
    events = {'predicted_energy': y[::,0::3].flatten(),
              'correct_energy': y_[::,0::3].flatten(), 
              
              'predicted_theta': np.mod(y[::,1::3],np.pi).flatten(),
              'correct_theta': y_[::,1::3].flatten(),
              
              'predicted_phi': np.mod(y[::,2::3],2*np.pi).flatten(),
              'correct_phi': y_[::,2::3].flatten()}
    
    figure, axes = plt.subplots(1,3, figsize=(13, 5))
    image = []
    image.append(axes[0].hist2d(events['correct_energy'], events['predicted_energy'], bins=bins, norm=LogNorm(), cmax = 1001))
    image.append(axes[1].hist2d(events['correct_theta'], events['predicted_theta'], bins=bins, norm=LogNorm(), cmax = 1001))
    image.append(axes[2].hist2d(events['correct_phi'], events['predicted_phi'], bins=bins, norm=LogNorm(), cmax = 1001))
    figure.tight_layout()
    return figure, axes, events


def plot_accuracy(history):
    """
    Learning curve; plot training and validation ACCURACY from model history
    
    Args:
        history : tf.keras.callbacks.History object returned from model.fit
    Returns:
        -
    Raises:
        TypeError : if history is not History object
    """
    if not isinstance(history, tf.keras.callbacks.History):
        raise TypeError('history must of type tf.keras.callbacks.History')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')


def plot_loss(history):
    """
    Learning curve; plot training and validation LOSS from model history
    
    Args:
        history : tf.keras.callbacks.History object returned from model.fit
    Returns:
        -
    Raises:
        TypeError : if history is not History object
    """
    if not isinstance(history, tf.keras.callbacks.History):
        raise TypeError('history must of type tf.keras.callbacks.History')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')


def plot_energy_distribution(data, bins=100, with_zeros=True):
    """
    Plots the energy distribution in given dataset. Obs: input data i expected
    with the format (energy, theta, phi).
    
    Args:
        data : predicted or label data
        bins : number of bins in histogram
        with_zeros : set False to omit zero energies
    Returns:
        -
    Raises:
        TypeError : if input data is not a numpy array
    """
    if not isinstance(data, np.ndarray):
        raise TypeError('label_data must be an numpy array.')
    
    energy = data[::,0::3].flatten()
    if not with_zeros:
        energy = np.setdiff1d(energy,[0])
    plt.hist(energy, bins, facecolor='blue', alpha=0.5)
    plt.show()
    
## ---------------------- Davids code -----------------------------------------
    
def plot_predictions_cart(prediction, labels, bins=500):
    """
    Use to plot a models predictions, with cartesian coordinates
    
    Args:
        prediction : use model.predict(data)
        labels : to compare with predictions
    Returns:
        figure, axes, events (dict)
    Raises : if data and labels is not the same length
        ValueError : if prediction and labels is not of same length
    """
    if not len(prediction)==len(labels):
        raise TypeError('The prediction must be of same length as labels.') 

    events = {'predicted_energy': prediction[::,0::4].flatten(),
              'correct_energy': labels[::,0::4].flatten(), 

              'predicted_x': prediction[::,1::4].flatten(),
              'correct_x': labels[::,1::4].flatten(),

              'predicted_y': prediction[::,2::4].flatten(),
              'correct_y': labels[::,2::4].flatten(),

              'predicted_z': prediction[::,3::4].flatten(),
              'correct_z': labels[::,3::4].flatten()}

    figure, axes = plt.subplots(1,4, figsize=(13, 5))
    image = []
    image.append(axes[0].hist2d(events['correct_energy'], events['predicted_energy'], bins=bins, norm=LogNorm(), cmax = 1001))
    image.append(axes[1].hist2d(events['correct_x'], events['predicted_x'], bins=bins, norm=LogNorm(), cmax = 1001))
    image.append(axes[2].hist2d(events['correct_y'], events['predicted_y'], bins=bins, norm=LogNorm(), cmax = 1001))
    image.append(axes[3].hist2d(events['correct_z'], events['predicted_z'], bins=bins, norm=LogNorm(), cmax = 1001))
    figure.tight_layout()
    return figure, axes, events
