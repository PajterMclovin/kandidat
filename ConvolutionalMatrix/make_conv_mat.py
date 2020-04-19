
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:16:50 2020

@author: david
"""

import numpy as np

from help_methods import neighbour_sorting_A, neighbour_sorting_D, get_crystals_of_type
      
### following matrices reduces and saves the convolutional matrices using CCT-sort
                            
def save_conv_mat_A():
    
    cluster_length = 16
    mat = np.zeros((162, 162*cluster_length), dtype=np.float32)
    
    crystals = get_crystals_of_type("A")
    
    for crystal in crystals:
        print("Generating neighbours to crystal:", crystal)
        neighbours = neighbour_sorting_A(crystal)
        for j in range(cluster_length):
                mat[neighbours[j]-1, (crystal-1)*cluster_length+j] = 1
    
    mat = reduce_columns(mat)   
    np.save('A_mat_CCT', mat)
    return

def save_conv_mat_D():
    
    cluster_length = 19
    mat = np.zeros((162, 162*cluster_length), dtype=np.float32)
    
    crystals = get_crystals_of_type("D")
    
    for crystal in crystals:
        print("Generating neighbours to crystal:", crystal)
        neighbours = neighbour_sorting_D(crystal)
        for j in range(cluster_length):
                mat[neighbours[j]-1, (crystal-1)*cluster_length+j] = 1
    
    mat = reduce_columns(mat)   
    np.save('D_mat_CCT', mat)
    return

def reduce_columns(m):
    """
    Returns a matrix containing the non-zero columns of the input matrix
    """
    s = np.sum(m, axis=0)
    c = np.array(np.nonzero(s))
    c = c[0,:]
    m_prime = m[:,c]
    
    return m_prime