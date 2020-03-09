#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 19:40:35 2019

@author: rickardkarlsson
"""

import make_conv_mat_AD as mat_AD
import numpy as np



if __name__=='__main__':
    np.save('conv_mat_C', mat_AD.get_conv_matrix_one_crystal_AD('C')) # saving the convolution matrix
