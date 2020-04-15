""" @PETER HALLDESTAM, 2020

    methods used to pre-process the data/label sets
    
"""
import numpy as np
from random import shuffle

from utils.help_methods import spherical_to_cartesian


def load_data(npz_file_name, total_portion, 
              cartesian=False, classification=False):
    """
    Reads a .npz-file containing simulation data in spherical coordinates and
    returns data/labels in spherical or cartesian coordinates in numpy arrays
    
    Args:
        npz_file_name : address string to .npz-file
        total_portion : the actual amount of total data to be used
        cartesian_coordinates : set True for cartesian coordinates
        classification_nodes : set True to train with classification nodes
    Returns:
        all data and labels in spherical (or cartesian) coordinates
    Raises:
        ValueError : if portion is not properly set
    """
    if not total_portion > 0 and total_portion <= 1:
        raise ValueError('total_portion must be in the interval (0,1].')
    data_set = np.load(npz_file_name)
    det_data = data_set['detector_data']
    labels = data_set['energy_labels']
    if cartesian:
        labels = spherical_to_cartesian(labels)
    if classification:
        labels = insert_classification_labels(labels, cartesian=cartesian)
    no_events = int(len(labels)*total_portion)
    print('Using {} events from {}'.format(no_events, npz_file_name))
    return det_data[:no_events], labels[:no_events]


def insert_classification_labels(labels, cartesian=False):
    """
    Inserts binary classification labels for each event as:
        energy==0 => mu=1
        energy >0 => mu=0
    """
    m = 3 + cartesian
    energy = labels[::,0::m]
    pos = labels[::,[i for i in range(len(labels[0])) if np.mod(i,m)!=0]]
    mu = (energy!=0)*1
    print(len(mu))
    max_mult = int(len(labels[0])/m)
    new_labels = np.zeros((len(labels), max_mult*(m+1)))
    new_labels[::,0::m+1] = mu
    new_labels[::,1::m+1] = energy
    new_labels[::,[i for i in range(len(labels[0])+max_mult) if np.mod(i,m+1)!=0 and np.mod(i-1,m+1)!=0]]=pos
    return new_labels
    
    
def get_eval_data(data, labels, eval_portion=0.1):
    """
    Detach final evaluation data.
    
    Args:
        data : dataset to split in two
        labels : corresponding data labels
        eval_portion : the amount of data for final evaluation
    Returns:
        training/validation data and labels
        final evaluation data and labels
    Raises:
        ValueError : if portion is not properly set
    """
    if not eval_portion > 0 and eval_portion < 1:
        raise ValueError('eval_portion must be in interval (0,1).')
    no_eval = int(len(data)*eval_portion)
    return data[no_eval:], labels[no_eval:], data[:no_eval], labels[:no_eval]


def sort_data(old_npz, new_npz):
    """
    Sort the label data matrix with rows (energy1, theta1, phi1, ... phiM), 
    where M is the max multiplicity.
    
    Args:
        old_npz : address of .npz-file with label data to sort
        new_npz : name of new .npz file
    """
    data_set = np.load(old_npz)
    labels = data_set['energy_labels']
    max_mult = int(len(labels[0])/3)
    energies = labels[::,::3]
    sort_indices = np.argsort(-energies) 
    sorted_labels = np.zeros([len(labels), 3*max_mult])
    for i in range(len(labels)):
        for j in range(max_mult):
            sorted_labels[i,3*j:3*(j+1)] = labels[i,3*sort_indices[i,j]:3*(sort_indices[i,j]+1)]
    np.savez(new_npz, detector_data=data_set['detector_data'], energy_labels=sorted_labels)


def add_empty_events(old_npz, new_npz, n):
    """
    Appends n empty events into data/label, shuffles them and then saves it to
    a new npz-file. Surely not the optimal solution but it should work. Maybe
    fucks up with larger data sets...
    
    Args:
        old_npz : address of .npz-file with label data to sort
        new_npz : name of new .npz file
    """
    data_set = np.load(old_npz)
    data = data_set['detector_data']
    labels = data_set['energy_labels']
    
    no_inputs = len(data[0])
    no_outputs = len(labels[0])
    data_list = np.vstack([data, np.zeros((n, no_inputs))]).tolist()
    label_list = np.vstack([labels, np.zeros((n, no_outputs))]).tolist()
    
    tmp = list(zip(data_list, label_list))
    shuffle(tmp)
    new_data, new_labels = zip(*tmp)
    np.savez(new_npz, detector_data=new_data, energy_labels=new_labels)
    
