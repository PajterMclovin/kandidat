# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:16:50 2020

@author: david
"""

import numpy as np

from help_methods import get_sorted_neighbours, get_crystals_of_type, rotate_orientation
      
### following matrices reduces and saves the convolutional matrices using CCT-sort


def save_conv_mat(crystal_type, use_rotations=False, use_reflections=False):
    
    name = ""
    no_rotations = 1
    
    if crystal_type == "A":
        cluster_length = 16
        if use_rotations:
            no_rotations = 5
            name = "_rot"
    elif crystal_type == "D":
        cluster_length = 19
        if use_rotations:
            no_rotations = 6
            name = "_rot"
    else:
        print("Invalid crystal type: ", crystal_type)
        return
    
    if use_reflections:
        refl_mult = 2
        name = name+"_refl"
    else:
        refl_mult = 1

    mat = np.zeros((162, 162*cluster_length*no_rotations*refl_mult), dtype=np.float32)
    crystals = get_crystals_of_type(crystal_type)
    
    for crystal in crystals:
        print("Generating neighbours to crystal:", crystal)
        # ccw
        neighbours = get_sorted_neighbours(crystal, crystal_type, "ccw")
        all_rotations = neighbours
        rotated_neighbours = neighbours
        for i in range(no_rotations-1):
            print("Rotating #",i)
            rotated_neighbours = rotate_orientation(rotated_neighbours, crystal_type)
            all_rotations=np.concatenate((all_rotations,rotated_neighbours))
        
        # cw
        if use_reflections:
            print("Flipping...")
            rotated_neighbours = get_sorted_neighbours(crystal, crystal_type, "cw")
            all_rotations=np.concatenate((all_rotations,rotated_neighbours))
            for i in range(no_rotations-1):
                print("Rotating #",i)
                rotated_neighbours = rotate_orientation(rotated_neighbours, crystal_type)
                all_rotations=np.concatenate((all_rotations,rotated_neighbours))
        
        for j in range(cluster_length*no_rotations*refl_mult):
            mat[all_rotations[j]-1, (crystal-1)*cluster_length*no_rotations*refl_mult+j] = 1
    
    mat = reduce_columns(mat)
    filename = crystal_type+"_mat_CCT"+name
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

