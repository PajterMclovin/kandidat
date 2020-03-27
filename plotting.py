""" @PETER HALLDESTAM, 2020
    
    Plot methods to analyze the neural network
    
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap

from utils import get_permutation_match

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

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
              
              'predicted_theta': y[::,1::3].flatten(),
              'correct_theta': y_[::,1::3].flatten(),
              
              'predicted_phi': np.mod(y[::,2::3], 2*np.pi).flatten(),
              'correct_phi': y_[::,2::3].flatten()}
    
    
    fig, axs = plt.subplots(1,3, figsize=(13, 5))
    colormap = truncate_colormap(plt.cm.afmhot, 0.0, 1.0)
    img = []
    img.append(axs[0].hist2d(events['correct_energy'], events['predicted_energy'], bins=bins, norm=LogNorm(), cmap=colormap, cmax = 1001))
    img.append(axs[1].hist2d(events['correct_theta'], events['predicted_theta'], bins=bins, norm=LogNorm(), cmap=colormap, cmax = 1001))
    img.append(axs[2].hist2d(events['correct_phi'], events['predicted_phi'], bins=bins, norm=LogNorm(), cmap=colormap, cmax = 1001))
    
    max_energy = 10
    max_theta = np.pi
    max_phi = 2*np.pi
    line_color = 'red'
    
    line = np.linspace(0,max_energy)
    for i in range(0,3):
        axs[i].plot(line,line, color=line_color, linewidth = 2, linestyle = '-.')

    text_size = 17
    axs[0].set_xlabel('Correct E [MeV]', fontsize = text_size)
    axs[1].set_xlabel('Correct \u03F4', fontsize = text_size)
    axs[2].set_xlabel('Correct \u03A6', fontsize = text_size)
    axs[0].set_ylabel('Reconstructed E [MeV]', fontsize = text_size)
    axs[1].set_ylabel('Reconstructed \u03F4', fontsize = text_size)
    axs[2].set_ylabel('Reconstructed E \u03A6', fontsize = text_size)
    
    
    axs[0].set_xlim([0, max_energy])
    axs[0].set_ylim([0, max_energy])
    axs[0].set_aspect('equal', 'box')
    axs[1].set_xlim([0, max_theta])
    axs[1].set_ylim([0, max_theta])
    axs[1].set_aspect('equal', 'box')
    axs[2].set_xlim([0, max_phi])
    axs[2].set_ylim([0, max_phi])
    axs[2].set_aspect('equal', 'box')

    cb1 = fig.colorbar(img[0][3], ax = axs[0], fraction=0.046, pad = 0.04)
    fig.delaxes(cb1.ax)
    cb2 = fig.colorbar(img[1][3], ax = axs[1], fraction=0.046, pad = 0.04)
    fig.delaxes(cb2.ax)
    cb3 = fig.colorbar(img[2][3], ax = axs[2], fraction=0.046, pad=0.04)

    cb1.ax.tick_params(labelsize = text_size)
    cb2.ax.tick_params(labelsize = text_size)
    cb3.ax.tick_params(labelsize = text_size)

    axs[0].tick_params(axis='both', which='major', labelsize=text_size)
    axs[1].tick_params(axis='both', which='major', labelsize=text_size)
    axs[2].tick_params(axis='both', which='major', labelsize=text_size)
    
    plt.sca(axs[0])
    plt.xticks(np.linspace(0, 10, 6),['0','2','4','6','8','10'])
    plt.yticks(np.linspace(0, 10, 6),['0','2','4','6','8','10'])
    plt.sca(axs[1])
    plt.xticks(np.linspace(0, np.pi, 3),['0','$\pi/2$','$\pi$'])
    plt.yticks(np.linspace(0, np.pi, 3),['0','$\pi/2$','$\pi$'])
    plt.sca(axs[2])
    plt.xticks(np.linspace(0, 2*np.pi, 3),['0','$\pi$','$2\pi$'])
    plt.yticks(np.linspace(0, 2*np.pi, 3),['0','$\pi$','$2\pi$'])
    
    fig.tight_layout()
    return fig, axs, events


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
