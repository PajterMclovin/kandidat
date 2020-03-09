#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:37:06 2019

@author: rickardkarlsson
"""

import make_conv_mat_AD as mat_AD
import numpy as np



if __name__=='__main__':
    np.save('conv_mat_A_rotated', mat_AD.get_conv_matrix_one_crystal_AD_rotations('A')) # saving the convolution matrix
    