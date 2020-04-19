# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:16:50 2020

@author: david
"""

import numpy as np

from help_methods import neighbour_sorting_A, neighbour_sorting_D, get_crystals_of_type, rotate_orientation
      
### following matrices reduces and saves the convolutional matrices using CCT-sort
                            
def save_conv_mat_A(use_rotations=False):
    
    cluster_length = 16
    if use_rotations:
        no_rotations = 5
        name = "_rot"
    else:
        no_rotations = 1
        name = ""
        
    mat = np.zeros((162, 162*cluster_length*no_rotations), dtype=np.float32)
    crystals = get_crystals_of_type("A")
    
    for crystal in crystals:
        print("Generating neighbours to crystal:", crystal)
        neighbours = neighbour_sorting_A(crystal)
        
        all_rotations = neighbours
        rotated_neighbours = neighbours
        for i in range(no_rotations-1):
            print("Rotating #",i+1)
            rotated_neighbours = rotate_orientation(rotated_neighbours, "A")
            all_rotations=np.concatenate((all_rotations,rotated_neighbours))
        
        for j in range(cluster_length*no_rotations):
                mat[all_rotations[j]-1, (crystal-1)*cluster_length*no_rotations+j] = 1
    
    mat = reduce_columns(mat)
    filename = "A_mat_CCT"+name
    print("Saving to ", filename)
    np.save(filename, mat)
    return mat

def save_conv_mat_D(use_rotations=False):
    
    cluster_length = 19
    if use_rotations:
        no_rotations = 6
        name = "_rot"
    else:
        no_rotations = 1
        name = ""
    mat = np.zeros((162, 162*cluster_length*no_rotations), dtype=np.float32)
    
    crystals = get_crystals_of_type("D")
    
    for crystal in crystals:
        print("Generating neighbours to crystal:", crystal)
        neighbours = neighbour_sorting_D(crystal)
        
        all_rotations = neighbours
        rotated_neighbours = neighbours
        for i in range(no_rotations-1):
            print("Rotating #",i+1)
            rotated_neighbours = rotate_orientation(rotated_neighbours, "D")
            all_rotations=np.concatenate((all_rotations,rotated_neighbours))
        
        for j in range(cluster_length*no_rotations):
                mat[all_rotations[j]-1, (crystal-1)*cluster_length*no_rotations+j] = 1
            
    mat = reduce_columns(mat)   
    filename = "D_mat_CCT"+name
    print("Saving to ", filename)
    np.save(filename, mat)
    return mat

def reduce_columns(m):
    """
    Returns a matrix containing the non-zero columns of the input matrix
    """
    s = np.sum(m, axis=0)
    c = np.array(np.nonzero(s))
    c = c[0,:]
    m_prime = m[:,c]
    
    return m_prime