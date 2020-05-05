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

def count_rec(evts, eps = 0.01):
    # räknar efter kvadrant..:
    # c1: finns, rek. till att finnas
    # c2: finns inte, rek. till att finnas
    # c3: finns inte, rek. till att inte finnas
    # c4: finns, rek. till att inte finnas
    # ger svaret i procent
    ce = evts['correct_energy']
    pe = evts['predicted_energy']
    tot_evts = len(ce)

    c1, c2, c3, c4 = 0, 0, 0, 0
    
    for i in range(len(ce)):
        if ce[i]>eps:
            if pe[i]>eps:
                c1 += 1
            else:
                c4 += 1
        if ce[i]<eps:
            if pe[i]>eps:
                c2 += 1
            else:
                c3 += 1
    proc = np.divide([c1, c2, c3, c4], tot_evts)*100
    return proc

def plot():
    titles = ["m5_1700k_200_F_64_32_10"]
    for title in titles: 
        h, mop, evts = load(title)
        plot_hist(h, title)
        plot_predictions_evt(evts, show_detector_angles=True, title=title)
    return
