""" @PETER HALLDESTAM, 2020
    
    Tensor transformations used in the network
    
"""

import numpy as np
import tensorflow.keras.backend as K
from math import factorial
from itertools import permutations

def get_permutation_tensor(n, m=3):
    """
    Returns a tensor containing all n! permutation matrices. Dimension: [n!, m*n, n*n]
    m is the no. outputs per reconstructed photon (4 in cartesian coordinates)
    """
    permutation_tensor = np.zeros((factorial(n), m*n, m*n))
    depth = 0
    for perm in permutations(range(n)):    
        for i in range(n):
            permutation_tensor[depth, m*i:m*(i+1):, m*perm[i]:m*(perm[i]+1):] = np.identity(m)
        depth += 1
    return K.constant(permutation_tensor)


def get_identity_tensor(n, m=3):
    """
    Returns a tensor containing n! identity matrices. Dimension: [n!, m*n, m*n]
    m is the no. outputs per reconstructed photon (4 in cartesian coordinates)
    """
    identity_tensor = np.zeros((factorial(n), m*n, m*n))
    for depth in range(factorial(n)):
            for i in range(n):
                identity_tensor[depth, m*i:m*(i+1), m*i:m*(i+1)] = np.identity(m)
    return K.constant(identity_tensor)

def get_shift_tensor(n, m=3):
    """
    Returns a tensor that rearranges the network output tensor from the form
    (energy, pos, ... mu,...) => (mu, energy, pos, ...)
    """
    shift_tensor = np.zeros((m*n, m*n))
    for i in range(n):
        shift_tensor[i-n, m*i] = 1
        shift_tensor[(m-1)*i:(m-1)*(i+1):, m*i+1:m*(i+1):] = np.identity(m-1)
    return K.constant(shift_tensor)
    
