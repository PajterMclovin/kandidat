# -*- coding: utf-8 -*-
"""
Created on Mon May  4 12:44:21 2020

@author: david
"""

import pickle 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap

from plotting import plot_predictions_evt

def load(foldername):
    h = 0
    with open('Resultat/'+foldername+'/traininghistory', 'rb') as f:
        h = pickle.load(f)
    mop = np.load('Resultat/'+foldername+'/mop.npy', allow_pickle=True).tolist()
    evts = np.load('Resultat/'+foldername+'/events.npy', allow_pickle=True).tolist()
    return h, mop, evts

def plot_hist(h, title):
    fig, axs = plt.subplots()
    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.grid()
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

titles = ["m5_1700k_200_F_64_32_10"]
for title in titles: 
    h, mop, evts = load(title)
    plot_hist(h, title)
    plot_predictions_evt(evts, show_detector_angles=True, title=title)
