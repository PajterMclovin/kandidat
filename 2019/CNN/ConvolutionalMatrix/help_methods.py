#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 19:34:50 2019

@author: rickardkarlsson
"""

#### METHODS FOR OTHER SCRIPTS IN CREATING CONVOLUTIONAL MATRICES THAT INCLUDE NEIGHBOURS #########

import numpy as np
import sympy as sp
neighbours_A = 16
neighbours_D = 19

nbr_of_A = 12
nbr_of_B = 60
nbr_of_C = 60
nbr_of_D = 30

def read_geom_txt():
    """" Reads the geometry file and removes white space """
    out=[]
    with open('geom_xb.txt', 'r') as f:         # Open file
        for lines in f:
            tmp=lines.split('(')[1]
            tmp=tmp.split(')')[0]
            tmp=tmp.replace(' ','')
            tmp=tmp.split(',')
            out.append(tmp)
    return out


## IMPORTANT VARIABLES
crystal_size = 5.5 # solid angle of a crystal
geom_data = read_geom_txt()


def get_crystal_type(crystal_index):
    """
        Get type of crystal from geom_xb.txt-file

        Input crystal_index(int): Index of crystal to get type of

        Output (string): Type of the crystal
    """
    return geom_data[crystal_index-1][1]

def get_closest_neighbours(crystal_index):
    """
        Returns the neighbours of a crystal

        Input
            crystal_number (int): the number representing a certain index from the geom_xb.txt file
            geom_data (string array): the geom_xb.txt file


        Output (string array): the selected neighbours
    """
    if crystal_index == -1:
        return
    index = int(crystal_index)-1
    out = geom_data[index][5:11]
    return np.asarray(out).astype(np.int16)

def get_second_closest_neighbours():
    """"
        Calculates the nearest and second nearest neighbours to each crystal

        Output (int array): a 162x23 array with the neighbours (sorted in ascending order)
                            where zeros mean that there is no neighbour there
    """

    max_nbr_of_neighbour = 23 # Width of neighbour-array so that the biggest neighbourhood will fit
    out = np.zeros(shape=(162,max_nbr_of_neighbour), dtype=np.int32) # Iniatilize output array

    for current_crystal in geom_data: # Loops through each crystal

        # Makes an array that checks which crystal are visited when checking neighbours
        visited = np.zeros(shape=(162+1,1), dtype=np.bool)
        current_crystal = int(current_crystal[0]) # cast string to int
        current_neighbours = get_closest_neighbours(current_crystal)

        for n in current_neighbours: # Loops through each neighbour to the current crystal in the outer loop
            second_closest_neighbours = get_closest_neighbours(n) # get closest neighbours to
                                                                            # the current neighbour
            if n != 0: # Checks if valid crystal
                visited[n] = True   # add the visited crystal to the array
                for c in second_closest_neighbours: # Loop each neighbour to the current visited neighbour
                    if c != 0: # Checks if valid crystal
                        visited[c] = True # add the visited crystal to the array

        visited = np.nonzero(visited)[0] # Indices for the visited crystal corresponds to crystal numbers
        out[current_crystal-1][0:len(visited)]=visited # Output is the visited crystals from each crystal

    return out



##############################################################################
### Geometric methods that calculates positions and angles ###################
##############################################################################
def projection_on_plane(vector, normal_vector):
    """
     Used for calculating angles in angle_between_two_crystals
    """
    # Initialize vectors
    ref_vector = normal_vector
    normal_vector = normal_vector/np.linalg.norm(normal_vector) # probably already normalized

    # Define a plane
    origin_point = sp.Point3D(ref_vector[0],ref_vector[1],ref_vector[2])
    normal_vector = (normal_vector[0],normal_vector[1],normal_vector[2])
    plane = sp.Plane(origin_point,normal_vector = normal_vector)

    # Project the crystal_index vector to the plane
    projected_point = sp.Point3D(vector[0],vector[1],vector[2])
    projection = plane.projection(projected_point)

    return np.array([projection[0],projection[1],projection[2]],dtype=np.float32)-ref_vector


def coordinates_XYZ_crystal(crystal_index):
    crystal_index -= 1
    theta=float(geom_data[crystal_index][2])*np.pi/180
    phi=float(geom_data[crystal_index][3])*np.pi/180

    return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)],dtype=np.float32)

def distance_between_two_crystals(index1,index2):
    """
        Calculates distance between two crystal.

        Output: (float) distance
    """
    r1 = coordinates_XYZ_crystal(index1)
    r2 = coordinates_XYZ_crystal(index2)
    return np.linalg.norm(r1-r2)

def angle_between_two_crystals(index1,index2,index_ref=None):
    """
        Calculates angle between two crystal in degrees.

        Output: (float) angle in radians
    """
    r1 = coordinates_XYZ_crystal(index1)
    r2 = coordinates_XYZ_crystal(index2)
    if index_ref == None:
        crossproduct = np.cross(r1,r2)
        abs_product = np.linalg.norm(r1)*np.linalg.norm(r2)
        angle = np.arcsin(np.linalg.norm(crossproduct)/abs_product)
        return angle
    else:
        r_ref = coordinates_XYZ_crystal(index_ref)
        r1 = projection_on_plane(r1-r_ref,r_ref)
        r2 = projection_on_plane(r2-r_ref,r_ref)
       
        crossproduct = np.cross(r1,r2)
        if np.array_equal(r1,r2):
            return 0
        else:
            r1=r1/np.linalg.norm(r1)
            r2=r2/np.linalg.norm(r2)
            prod = np.dot(r1,r2)
            
            if prod > 1:
                print("angle_between_two_crystals product out of bounds, set to [-1, 1]")
                print("real product: ")
                print(prod)
                prod = 1
            if prod < -1:
                print("angle_between_two_crystals product out of bounds, set to [-1, 1]")
                print("real product: ")
                print(prod)
                prod = -1
            
            angle = np.arccos(np.dot(r1,r2))

            if np.linalg.norm(crossproduct+r_ref)<1:
                sign = -1
            else:
                sign = 1
            return sign*angle*180/np.pi

# This index finds which of the crystals that is closest to the beam-out crystal (crystal 81)
# MOD by david, defaults to 81 but can be given other crystal
def find_index_shortest_distance_to_crystal(crystal_array, ref_crystal=81):
    tmp_distances=np.ones(len(crystal_array),dtype=np.float32)
    for i in range(len(crystal_array)):
        tmp_distances[i]=distance_between_two_crystals(ref_crystal,crystal_array[i])
    return crystal_array[tmp_distances.argmin()]





##########################################################
########### Orientation methods ##########################
##########################################################

def rotate_orientation(input_neighbourhood, crystal_type):
    """
        Used to rotate a neighbourhood
    """
    if crystal_type == 'A':
        first_layer_size = 5
    elif crystal_type == 'D':
        first_layer_size = 6
    else:
        raise ValueError('Wrong type of crystal, your input type was: ',crystal_type)

    mid_crystal = input_neighbourhood[0]
    first_layer = np.asarray(input_neighbourhood[1:first_layer_size+1])
    second_layer = np.asarray(input_neighbourhood[first_layer_size+1:])
    first_layer = np.roll(first_layer,1)
    second_layer = np.roll(second_layer,2)
    output = np.concatenate((first_layer,second_layer))
    return np.insert(output,0,mid_crystal)


def correct_orientation_with_angles(mid_crystal, neighbour_layers, ref_crystal=81):
    closest_to_ref = find_index_shortest_distance_to_crystal(neighbour_layers, ref_crystal)
    angles = [] # Add angles to this list, they will have the same index as corresponding crystal
    angle_index_map = {}
    for crystal in neighbour_layers:
        angle = angle_between_two_crystals(closest_to_ref,crystal,mid_crystal)
        angles.append(angle)
        angle_index_map.update({angle:crystal})
    angles.sort()
    start_index = angles.index(0)
    out_angles = []
    for i in range(len(angles)):
        i = (start_index+i)%len(angles)
        angle = angle_index_map[angles[i]]
        out_angles.append(angle)
    return np.asarray(out_angles,dtype=np.int32)



def correct_orientation_first_neighbours(input_neighbourhood):
    """
        Only works for first layer and is taken from tutorial-code
    """
    neighbour_size = len(input_neighbourhood)
    theta=np.zeros(neighbour_size,dtype=np.float32)
    phi = np.zeros(neighbour_size, dtype=np.float32)

    for j in range(len(input_neighbourhood)):
        crystal = input_neighbourhood[j]
        if crystal != -1:
            theta[j]=np.array(geom_data[crystal-1][2]).astype(np.float32)
            phi[j]=np.array(geom_data[crystal-1][3]).astype(np.float32)
    theta_max_index=theta.argmax()

    max_theta=theta[theta_max_index]
    theta[theta_max_index]=0
    theta_second_largest_index=theta.argmax()
    theta_second_largest=theta[theta_second_largest_index]


    if max_theta-theta_second_largest<5.5:
        if phi[theta_max_index]>phi[theta_second_largest_index] and abs(phi[theta_second_largest_index]-phi[theta_max_index])<180:
            start_crystal=theta_max_index
        else:
            start_crystal=theta_second_largest_index
    else:
        start_crystal = theta_max_index
    if start_crystal==0:
        neighbour_oriented=input_neighbourhood
    else:
        neighbour_oriented=np.concatenate((input_neighbourhood[start_crystal:],input_neighbourhood[0:start_crystal]))

    return np.array(neighbour_oriented).astype(np.int)
