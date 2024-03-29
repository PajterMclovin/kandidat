""" @PETER HALLDESTAM, 2020
    
    Plot methods to analyze the neural network
    
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap

from utils import get_detector_angles


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot_predictions_evt(events, bins=500, show_detector_angles=False, title = ""):
    """
    Use to plot a models predictions in similar format as previous years, i.e. 2d histograms ("lasersvärd")
    
    Args:
        prediction : use model.predict(data)
        labels : to compare with predictions
        bins : number of bins i histogram
        permutation : True if network used permutational loss, False if not
        cartesian_coordinate : True if network is trained with cartesin coordinates
        loss_type : used to match permutations
        show_detector_angles : shows points representing the 162 XB detector crystals
        show_description : adds a descriptive title
    Returns:
        figure, axes, events (dict)
    Raises : if data and labels is not the same length
        ValueError : if prediction and labels is not of same length
    """
       
    fig, axs = plt.subplots(1,3, figsize=(20, 8))
    colormap = truncate_colormap(plt.cm.copper, 0.0, 1.0)
    img = []
    img.append(axs[0].hist2d(events['correct_energy'], events['predicted_energy'],cmap=colormap, bins=bins, norm=LogNorm()))
    img.append(axs[1].hist2d(events['correct_theta'], events['predicted_theta'], cmap=colormap, bins=bins, norm=LogNorm()))
    img.append(axs[2].hist2d(events['correct_phi'], events['predicted_phi'], cmap=colormap, bins=bins, norm=LogNorm()))
    
    max_energy = 10
    max_theta = np.pi
    max_phi = 2*np.pi
    line_color = 'blue'
    ml = -0.15
    min_e = 0
    min_t = 0
    min_p = 0
    
    
    line = np.linspace(0,max_energy)
    for i in range(0,3):
        axs[i].plot(line,line, color=line_color, linewidth = 2, linestyle = '-.')

    if show_detector_angles:
        detector_theta, detector_phi = get_detector_angles()
        axs[1].scatter(detector_theta, detector_theta, marker='x')
        axs[2].scatter(detector_phi, detector_phi, marker='x')
    
    text_size = 17
    axs[0].set_xlabel('Correct E [MeV]', fontsize = text_size)
    axs[1].set_xlabel('Correct \u03F4', fontsize = text_size)
    axs[2].set_xlabel('Correct \u03A6', fontsize = text_size)
    axs[0].set_ylabel('Reconstructed E [MeV]', fontsize = text_size)
    axs[1].set_ylabel('Reconstructed \u03F4', fontsize = text_size)
    axs[2].set_ylabel('Reconstructed E \u03A6', fontsize = text_size)
    
    
    axs[0].set_xlim([-2, max_energy])
    axs[0].set_ylim([-2, max_energy])
    axs[0].set_aspect('equal', 'box')
    axs[1].set_xlim([min_t, max_theta])
    axs[1].set_ylim([min_t, max_theta])
    axs[1].set_aspect('equal', 'box')
    axs[2].set_xlim([min_p, max_phi])
    axs[2].set_ylim([min_p, max_phi])
    axs[2].set_aspect('equal', 'box')

    cb1 = fig.colorbar(img[0][3], ax = axs[0], fraction=0.046, pad = 0.04)
    # fig.delaxes(cb1.ax)
    cb2 = fig.colorbar(img[1][3], ax = axs[1], fraction=0.046, pad = 0.04)
    # fig.delaxes(cb2.ax)
    cb3 = fig.colorbar(img[2][3], ax = axs[2], fraction=0.046, pad=0.04)

    cb1.ax.tick_params(labelsize = text_size)
    cb2.ax.tick_params(labelsize = text_size)
    cb3.ax.tick_params(labelsize = text_size)

    axs[0].tick_params(axis='both', which='major', labelsize=text_size)
    axs[1].tick_params(axis='both', which='major', labelsize=text_size)
    axs[2].tick_params(axis='both', which='major', labelsize=text_size)
    
    #plt.title(title) 
    plt.sca(axs[0])
    plt.xticks(np.linspace(0, 10, 6),['0','2','4','6','8','10'])
    plt.yticks(np.linspace(0, 10, 6),['0','2','4','6','8','10'])
    
    plt.sca(axs[1])
    plt.title(title) 
    plt.xticks(np.linspace(0, np.pi, 3),['0','$\pi/2$','$\pi$'])
    plt.yticks(np.linspace(0, np.pi, 3),['0','$\pi/2$','$\pi$'])
    
    plt.sca(axs[2])
    plt.xticks(np.linspace(0, 2*np.pi, 3),['0','$\pi$','$2\pi$'])
    plt.yticks(np.linspace(0, 2*np.pi, 3),['0','$\pi$','$2\pi$'])
    
    
    
    fig.tight_layout()
    return fig, events

def plot_predictions(y, y_, bins=500, show_detector_angles=False):

    if not len(y)==len(y_):
        raise TypeError('The prediction must be of same length as labels.') 
                      
    events = {'predicted_energy': y[::,0::3].flatten(),
              'correct_energy': y_[::,0::3].flatten(), 
              
              'predicted_theta': y[::,1::3].flatten(),
              'correct_theta': y_[::,1::3].flatten(),
              
              'predicted_phi': np.mod(y[::,2::3], 2*np.pi).flatten(),
              'correct_phi': y_[::,2::3].flatten()}
    
    return plot_predictions_evt(events, bins, show_detector_angles)


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
        
    fig, axs = plt.subplots()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.grid()
    return fig


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
    

   
