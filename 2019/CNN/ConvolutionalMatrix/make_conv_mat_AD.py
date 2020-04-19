#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 19:40:35 2019

@author: rickardkarlsson
"""

import help_methods as hm
import numpy as np


## VARIABLES


## Load two layers of neighbours, nearest and second nearest.
neighbours_including_second_nearest = hm.neighbours_including_second_nearest

def count_crystal_types():
    count_A = 0
    count_B = 0
    count_C = 0
    count_D = 0
    list = get_correct_type_crystal_AD()
    for crystal in list:
        if hm.get_crystal_type(crystal) == 'A':
            count_A += 1
        if hm.get_crystal_type(crystal) == 'B':
            count_B += 1
        if hm.get_crystal_type(crystal) == 'C':
            count_C += 1
        if hm.get_crystal_type(crystal) == 'D':
            count_D += 1
    print(count_A)
    print(count_B)
    print(count_C)
    print(count_D)

def get_correct_type_crystal_AD():
    """
        Returns index of type A and D crystals in a array
    """
    correct_crystals = []
    for crystal in range(len(hm.geom_data)):
        crystal +=1
        correct_crystals.append(crystal)
    return np.asarray(correct_crystals)

def first_layer_neighbours_AD(crystal_index):
    """
        Get neighbours for type-A or type-D crystal
    """
    if hm.get_crystal_type(crystal_index) == 'A':
        return hm.get_closest_neighbours(crystal_index)[0:5]
    elif hm.get_crystal_type(crystal_index) == 'B':
        return hm.get_closest_neighbours(crystal_index)[0:6]
    elif hm.get_crystal_type(crystal_index) == 'C':
        return hm.get_closest_neighbours(crystal_index)[0:6]
    elif hm.get_crystal_type(crystal_index) == 'D':
        return hm.get_closest_neighbours(crystal_index)[0:6]
    else:
        raise ValueError('Wrong input, not valid crystal')

def second_layer_neighbours_AD(crystal_index):
    """
        Returns first and second layer of neighbours to the crystal
    """
    first_layer_neighbour = first_layer_neighbours_AD(crystal_index)

    # Appends zero and crystal_index which is to be removed from array
    first_layer_neighbour = np.append(first_layer_neighbour,np.array([crystal_index, 0]))
    first_and_second_layer = neighbours_including_second_nearest[crystal_index-1]

    # Returns difference of first+second layer array and first array ---> only second layer neighbour
    return np.setdiff1d(first_and_second_layer,first_layer_neighbour)


def get_conv_matrix_AD():
    all_crystals = get_correct_type_crystal_AD()
    out = np.zeros((162, 162*19), dtype=np.float32)
    for crystal in all_crystals:
        print("Generating neighbours to crystal:",crystal)
        first_layer = hm.correct_orientation_first_neighbours(first_layer_neighbours_AD(crystal))
        second_layer = hm.correct_orientation_with_angles(crystal,second_layer_neighbours_AD(crystal))
        if hm.get_crystal_type(crystal) == 'A':
            neighbours = np.concatenate((first_layer, np.array([-1]), second_layer))
        else: # if type is equal to 'D'
            neighbours = np.concatenate((first_layer, second_layer))
        for index_i in range(len(neighbours)):
            if neighbours[index_i] == -1:
                neighbours = np.delete(neighbours, index_i)
                neighbours = np.concatenate((neighbours, np.array([-1])))
        final_neighbour = np.concatenate((neighbours, np.array([crystal-1])))
        for index_j in range(len(final_neighbour)):
            if final_neighbour[index_j] != -1:
                out[final_neighbour[index_j]-1, (crystal-1)*len(final_neighbour) +index_j] = 1
    return out

def get_conv_matrix_one_crystal_AD(crystal_type):
    if crystal_type == 'A':
        neighbour_size = 16
    elif crystal_type == 'B':
        neighbour_size = 19
    elif crystal_type == 'C':
        neighbour_size = 19
    elif crystal_type == 'D':
        neighbour_size = 19
    all_crystals = get_correct_type_crystal_AD()
    out = np.zeros((162, 162*neighbour_size), dtype=np.float32)
    for crystal in all_crystals:
        if hm.get_crystal_type(crystal) != crystal_type:
            pass
        else:
            print("Generating neighbours to crystal:",crystal)
            first_layer = hm.correct_orientation_first_neighbours(first_layer_neighbours_AD(crystal))
            second_layer = hm.correct_orientation_with_angles(crystal,second_layer_neighbours_AD(crystal))
            neighbours = np.concatenate((np.array([crystal]),first_layer, second_layer))
            for index_j in range(len(neighbours)):
                out[neighbours[index_j]-1, (crystal-1)*len(neighbours) +index_j] = 1

    """
    Used to remove zero columns, works well for A & D but not for B & C
    out = np.transpose(out)
    out = out[~(out==0).all(1)]
    out = np.transpose(out)
    """
    return out

def get_conv_matrix_one_crystal_AD_fix(crystal_type):
    if crystal_type == 'A':
        neighbour_size = 16
    elif crystal_type == 'B':
        neighbour_size = 19
    elif crystal_type == 'C':
        neighbour_size = 19
    elif crystal_type == 'D':
        neighbour_size = 19
    all_crystals = get_correct_type_crystal_AD()
    out = np.zeros((162, 162*neighbour_size), dtype=np.float32)
    for crystal in all_crystals:
        if hm.get_crystal_type(crystal) != crystal_type:
            pass
        else:
            print("Generating neighbours to crystal:",crystal)
            first_layer = hm.correct_orientation_first_neighbours(first_layer_neighbours_AD(crystal))
            second_layer = hm.correct_orientation_with_angles(crystal,second_layer_neighbours_AD(crystal),
                                                              ref_crystal=first_layer[0])
            
            neighbours = np.concatenate((np.array([crystal]),first_layer, second_layer))
            for index_j in range(len(neighbours)):
                out[neighbours[index_j]-1, (crystal-1)*len(neighbours) +index_j] = 1

    """
    Used to remove zero columns, works well for A & D but not for B & C
    out = np.transpose(out)
    out = out[~(out==0).all(1)]
    out = np.transpose(out)
    """
    return out

def get_conv_matrix_one_crystal_AD_rotations(crystal_type):
    if crystal_type == 'A':
        neighbour_size = 16
        no_of_rotations = 5
    elif crystal_type == 'D':
        neighbour_size = 19
        no_of_rotations = 6
    all_crystals = get_correct_type_crystal_AD()
    out = np.zeros((162, 162*neighbour_size*no_of_rotations), dtype=np.float32)
    for crystal in all_crystals:
        if hm.get_crystal_type(crystal) != crystal_type:
            pass
        else:
            print("Generating neighbours to crystal:",crystal)
            first_layer = hm.correct_orientation_first_neighbours(first_layer_neighbours_AD(crystal))
            second_layer = hm.correct_orientation_with_angles(crystal,second_layer_neighbours_AD(crystal))
            neighbours = np.concatenate((first_layer, second_layer))
            neighbours = np.insert(neighbours,0,crystal)

            all_rotations = neighbours
            rotated_neighbours = neighbours
            for i in range(no_of_rotations-1):
                rotated_neighbours = hm.rotate_orientation(rotated_neighbours, crystal_type)
                all_rotations=np.concatenate((all_rotations,rotated_neighbours))
            for index_j in range(len(all_rotations)):
                out[all_rotations[index_j]-1, (crystal-1)*len(all_rotations) +index_j] = 1
    return out


def get_conv_matrix_one_crystal_AD_update(crystal_type):
    if crystal_type == 'A':
        neighbour_size = 16
        nbr_of_crystals = 12
    elif crystal_type == 'D':
        neighbour_size = 19
        nbr_of_crystals = 30
    all_crystals = get_correct_type_crystal_AD()
    out = np.zeros((162, nbr_of_crystals*neighbour_size), dtype=np.float32)
    count = 0
    for crystal in all_crystals:
        if hm.get_crystal_type(crystal) != crystal_type:
            pass
        else:
            print("Generating neighbours to crystal:",crystal)
            first_layer = hm.correct_orientation_first_neighbours(first_layer_neighbours_AD(crystal))
            second_layer = hm.correct_orientation_with_angles(crystal,second_layer_neighbours_AD(crystal))
            neighbours = np.concatenate((first_layer, second_layer))
            neighbours = np.insert(neighbours,0,crystal)
            index = 0
            for n in neighbours:
                out[n-1, count*len(neighbours)+index] = 1
                index += 1
            count += 1
    return out
