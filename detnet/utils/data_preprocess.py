""" @PETER HALLDESTAM, 2020

    methods used to pre-process the data/label sets
    
"""
import numpy as np
from numpy.random import permutation
from random import sample

from .help_methods import spherical_to_cartesian


def load_data(npz_file, total_portion, add_zeros=0, classification=False, hinge=False):
    """
    Reads a .npz-file from the address string: npz_file that contains simulation 
    data in spherical coordinates. I transforms to cartesian coordinates and
    may add n empty events and also add classification nodes.
    
    """
    if not total_portion > 0 and total_portion <= 1:
        raise ValueError('total_portion must be in the interval (0,1].')
        
    data_set = np.load(npz_file)
    det_data = data_set['detector_data']
    labels = spherical_to_cartesian(data_set['energy_labels'])
    
    if add_zeros:
        det_data, labels = insert_empty_events(det_data, labels, n=add_zeros)
    
    if classification:
        labels = insert_classification_labels(labels, hinge)
    
    no_events = int(len(labels)*total_portion)
    print('Using {} events from {}'.format(no_events, npz_file))
    return det_data[:no_events], labels[:no_events]


def insert_empty_events(data, labels, n):
    """
    Inserts n empty events into given data and label set.
    
    """
    if n > 10*len(labels):
        raise ValueError('Too many zeros to add...')
    
    print('inserting {} zero events...'.format(n))
    new_data = data
    new_labels = labels
    no_insertions = n
    
    empty_data = np.zeros(data[0].shape)
    empty_label = np.zeros(labels[0].shape)
    
    while no_insertions > len(new_labels):
        insert_indices = permutation(len(new_labels))
        new_data = np.insert(new_data, insert_indices, empty_data, axis=0)
        new_labels = np.insert(new_labels, insert_indices, empty_label, axis=0)
        no_insertions -= len(insert_indices)
    
    insert_indices = permutation(no_insertions)
    new_data = np.insert(new_data, insert_indices, empty_data, axis=0)
    new_labels = np.insert(new_labels, insert_indices, empty_label, axis=0)
    return new_data, new_labels
    

def insert_classification_labels(labels, hinge):
    """
    Inserts binary classification labels (b,px,py,pz) for each event as:
        energy==0 => b=1
        energy>0 => b=0
    """
    px_ = labels[::,0::3].flatten()
    py_ = labels[::,1::3].flatten()
    pz_ = labels[::,2::3].flatten()
    
    energy = np.sqrt(px_*px_+py_*py_+pz_*pz_)
    b_ = np.array((energy!=0)*1)
    
    if hinge:
        b_ = 2*(b_ - .5)
    
    max_mult = int(len(labels[0])/3)
    return np.reshape(np.column_stack((b_, px_, py_, pz_)), (-1, 4*max_mult))
    
 
def get_eval_data(data, labels, eval_portion=0.1):
    """
    Seperates into data and labels set into one for training/validiation and
    another for final evaluation.

    """
    if not eval_portion > 0 and eval_portion < 1:
        raise ValueError('eval_portion must be in interval (0,1).')
    no_eval = int(len(data)*eval_portion)
    return data[no_eval:], labels[no_eval:], data[:no_eval], labels[:no_eval]


def sort_data(old_npz, new_npz):
    """
    Sort the label data matrix with rows (energy1, theta1, phi1, ... phiM), 
    where M is the max multiplicity and saves to new npz file.

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


    
    
    
