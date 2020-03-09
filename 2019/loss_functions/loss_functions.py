import numpy as np
import tensorflow as tf
import itertools as it

#####################################################
################# LOSS FUNCTIONS ####################
#####################################################

'''The standard cost function, testing all different combinations of paring of the gamma-rays.
y = network output
y_ = correct labels corresponding to the network output
num_splits = the maximum multiplicity of the events
lam_X = weighting factors
E_offset = the energy offset used for relatice loss
theta_offset/phi_offset = do not use, left from beginning of the project
use_relative_loss = if relative loss is going to be used; 1 = yes and 0 = no
weight_decay_loss = the weight decay loss (returned from the hidden layer method)
beta = weighting factor for weight decay loss (probably do not use; 0)

*Can be used for both relativistic (y and y_ in same frame (beam/lab)) and non relativistic data
'''

def energy_theta_phi_permutation_loss(y, y_, num_splits, lam_E = 1, lam_theta = 1, lam_phi = 1, E_offset = 0.1, theta_offset = 0.1, phi_offset = 0.1, use_relative_loss = 0, weight_decay_loss = 0, beta = 0):
    #Dividing into individual gamma-ray blocks
    splited_y = tf.split(y, num_splits, axis=1)
    splited_y_ = tf.split(y_, num_splits, axis=1)
    temp_shape = tf.shape(tf.split(splited_y[0], 3, axis=1))

    #Calculates the loss for ONE of the possible permutations of the pairing
    def one_comb_loss(splited_y, splited_y_, index_list):
        temp = tf.zeros(temp_shape, dtype=tf.float32)
        if use_relative_loss == 1:
            #Relative loss
            for i in range(len(index_list)):
                E, theta, phi = tf.split(splited_y[i], 3, axis=1)
                E_, theta_, phi_ = tf.split(splited_y_[index_list[i]], 3, axis=1)

                tmp_loss_E = lam_E*tf.square(tf.divide(E-E_, E_+E_offset))
                tmp_loss_theta = lam_theta*tf.square(theta-theta_)
                tmp_loss_phi = lam_phi*tf.square(tf.mod(phi - phi_ + np.pi, 2*np.pi) - np.pi)

                temp = temp + tmp_loss_E + tmp_loss_theta + tmp_loss_phi
        else:
            #Absolute loss
            for i in range(len(index_list)):
                E, theta, phi = tf.split(splited_y[i], 3, axis=1)
                E_, theta_, phi_ = tf.split(splited_y_[index_list[i]], 3, axis=1)

                tmp_loss_E = lam_E*tf.square(E-E_)
                tmp_loss_theta = lam_theta*tf.square(theta-theta_)
                tmp_loss_phi = lam_phi*tf.square(tf.mod(phi - phi_ + np.pi, 2*np.pi) - np.pi)

                temp = temp + tmp_loss_E + tmp_loss_theta + tmp_loss_phi
        return temp

    #Loops over every possible pairing permutation and calculates the losses with one_comb_loss, gathered losses in list_of_tensors
    list_of_tensors = [one_comb_loss(splited_y, splited_y_, index_list) for index_list in it.permutations(range(num_splits), num_splits)]

    #Initialize infinite loss and then iterate over all the losses to find the lowest
    loss = tf.divide(tf.constant(1, dtype=tf.float32), tf.zeros(temp_shape, dtype=tf.float32))
    for i in range(len(list_of_tensors)):
        loss = tf.minimum(loss, list_of_tensors[i])
    return tf.reduce_mean(loss) + beta*weight_decay_loss

#Similar to the above cost function; sorting the network output for the evaluation data set, finding the pairing with the lowest loss.
#The difference is that here the data is not TensorFlow tensors (damn you tensors!!!) and is therefore much more easy to work with
#Splits the output in individual energy and angle quantities, gathered in events
def network_permutation_sort(y, y_, lam_E = 1, lam_theta = 1, lam_phi = 1, E_offset = 0.1, theta_offset = 0.1, phi_offset = 0.1, use_relative_loss = 0):
    E, theta, phi, E_label, theta_label, phi_label = [], [], [], [], [], []

    def one_event_permutation_sort(event_y, event_y_):
        index_min_loss = False
        tmp_min_loss = np.inf
        for index_list in it.permutations(range(int(len(event_y)/3)), int(len(event_y)/3)):
            tmp_loss = 0
            if use_relative_loss == 0:
                #Absolute loss
                for i in range(int(len(event_y)/3)):
                    tmp_loss_E = lam_E*np.power((event_y[3*i] - event_y_[3*index_list[i]]), 2)
                    tmp_loss_theta = lam_theta*np.power(event_y[3*i+1] - event_y_[3*index_list[i]+1], 2)
                    tmp_loss_phi = lam_phi*np.power(np.mod(event_y[3*i+2] - event_y_[3*index_list[i]+2] + np.pi, 2*np.pi)-np.pi, 2)

                    tmp_loss = tmp_loss + tmp_loss_E + tmp_loss_theta + tmp_loss_phi
            else:
                #Relative loss
                for i in range(int(len(event_y)/3)):
                    tmp_loss_E = lam_E*np.power((event_y[3*i] - event_y_[3*index_list[i]])/(event_y_[3*index_list[i]] + E_offset), 2)
                    tmp_loss_theta = lam_theta*np.power((event_y[3*i+1] - event_y_[3*index_list[i]+1]), 2)
                    tmp_loss_phi = lam_phi*np.power(np.mod(event_y[3*i+2] - event_y_[3*index_list[i]+2] + np.pi, 2*np.pi)-np.pi, 2)

                    tmp_loss = tmp_loss + tmp_loss_E + tmp_loss_theta + tmp_loss_phi
            if tmp_loss < tmp_min_loss:
                tmp_min_loss = tmp_loss
                index_min_loss = index_list
        return index_min_loss

    for i in range(len(y)):
        index_min_loss = one_event_permutation_sort(y[i], y_[i])
        for j in range(int(len(y[0])/3)):
            E.append(y[i][3*j])
            theta.append(y[i][3*j+1])
            phi.append(y[i][3*j+2])
            E_label.append(y_[i][3*index_min_loss[j]])
            theta_label.append(y_[i][3*index_min_loss[j]+1])
            phi_label.append(y_[i][3*index_min_loss[j]+2])

    events = {0: E,
              1: E_label,
              2: theta,
              3: theta_label,
              4: phi,
              5: phi_label}
    return events

'''Similar as energy_theta_phi_permutation_loss (relative loss not implemented)
*Applies the Doppler correction to y (lab frame --> beam frame)
*Observe that y (network output) should be in the lab_frame while y_ (correct label data) should be in beam_frame
*Used for a network trained to optimize the energy reconstruction in the beam frame
*Observe: The network still outputs the energy (and theta/phi) in lab frame and therefore has to be corrected thereafter'''
def energy_theta_phi_permutation_loss_lab_E_to_beam_E(y, y_, num_splits, lam_E = 1, lam_theta = 1, lam_phi = 1, E_offset = 0.1, theta_offset = 0.1, phi_offset = 0.1, use_relative_loss = 0, weight_decay_loss = 0, beta = 0, rel_beta = 0.7):
    splited_y = tf.split(y, num_splits, axis=1)
    splited_y_ = tf.split(y_, num_splits, axis=1)
    temp_shape = tf.shape(tf.split(splited_y[0], 3, axis=1))
    use_relative_loss = 0       #Did not implemented relative loss

    def one_comb_loss(splited_y, splited_y_, index_list):
        temp = tf.zeros(temp_shape, dtype=tf.float32)
        if use_relative_loss == 1:
            #Relative loss (not implemented)
            temp = 0
        else:
            #Absolute loss
            for i in range(len(index_list)):
                E, theta, phi = tf.split(splited_y[i], 3, axis=1)
                E_, theta_, phi_ = tf.split(splited_y_[index_list[i]], 3, axis=1)

                E_prim = tf.multiply(tf.divide(1 - tf.multiply(rel_beta, tf.cos(theta)), tf.sqrt(1 - tf.multiply(rel_beta, rel_beta))), E)

                tmp_loss_E = lam_E*tf.square(E_prim-E_)
                tmp_loss_theta = lam_theta*tf.square(theta-theta_)
                tmp_loss_phi = lam_phi*tf.square(tf.mod(phi - phi_ + np.pi, 2*np.pi) - np.pi)

                temp = temp + tmp_loss_E + tmp_loss_theta + tmp_loss_phi
        return temp

    # All losses in a list. Inner loop over all permutations.
    list_of_tensors = [one_comb_loss(splited_y, splited_y_, index_list) for index_list in it.permutations(range(num_splits), num_splits)]

    loss = tf.divide(tf.constant(1, dtype=tf.float32), tf.zeros(temp_shape, dtype=tf.float32))
    for i in range(len(list_of_tensors)):
        loss = tf.minimum(loss, list_of_tensors[i])
    return tf.reduce_mean(loss) + beta*weight_decay_loss

'''Similar as network_permutation_sort (relative loss not implemented)
*Applies the Doppler correction to y (lab frame --> beam frame)
*Observe that y (network output) should be in the lab_frame while y_ (correct label data) should be in beam_frame
*Used for a network trained to optimize the energy reconstruction in the beam frame'''
def network_permutation_sort_lab_E_to_beam_E(y, y_, lam_E = 1, lam_theta = 1, lam_phi = 1, E_offset = 0.1, theta_offset = 0.1, phi_offset = 0.1, use_relative_loss = 0, rel_beta = 0.7):
    E, theta, phi, E_label, theta_label, phi_label = [], [], [], [], [], []

    def one_event_permutation_sort(event_y, event_y_):
        index_min_loss = False
        tmp_min_loss = np.inf
        use_relative_loss = 0       #Relative loss not implemented
        for index_list in it.permutations(range(int(len(event_y)/3)), int(len(event_y)/3)):
            tmp_loss = 0
            if use_relative_loss == 0:
                #Absolute loss
                for i in range(int(len(event_y)/3)):
                    E_prim = ((1-rel_beta*np.cos(event_y[3*i+1]))/(np.sqrt(1 - rel_beta*rel_beta)))*event_y[3*i]

                    tmp_loss_E = lam_E*np.power((E_prim - event_y_[3*index_list[i]]), 2)
                    tmp_loss_theta = lam_theta*np.power(event_y[3*i+1] - event_y_[3*index_list[i]+1], 2)
                    tmp_loss_phi = lam_phi*np.power(np.mod(event_y[3*i+2] - event_y_[3*index_list[i]+2] + np.pi, 2*np.pi)-np.pi, 2)

                    tmp_loss = tmp_loss + tmp_loss_E + tmp_loss_theta + tmp_loss_phi
            else:
                #Relative loss
                tmp_loss = 0
            if tmp_loss < tmp_min_loss:
                tmp_min_loss = tmp_loss
                index_min_loss = index_list
        return index_min_loss

    for i in range(len(y)):
        index_min_loss = one_event_permutation_sort(y[i], y_[i])
        for j in range(int(len(y[0])/3)):
            E.append(y[i][3*j])
            theta.append(y[i][3*j+1])
            phi.append(y[i][3*j+2])
            E_label.append(y_[i][3*index_min_loss[j]])
            theta_label.append(y_[i][3*index_min_loss[j]+1])
            phi_label.append(y_[i][3*index_min_loss[j]+2])

    events = {0: E,
              1: E_label,
              2: theta,
              3: theta_label,
              4: phi,
              5: phi_label}
    return events

'''Similar as energy_theta_phi_permutation_loss (relative loss may be correctly implemented (do not remember...))
*Classification node implemented with extra weighting factor lam_detect
'''
def energy_theta_phi_permutation_loss_quattro(y, y_, num_splits, lam_E = 1, lam_theta = 1, lam_phi = 1, lam_detect = 1, E_offset = 0.1, theta_offset = 0.1, phi_offset = 0.1, use_relative_loss = 0, weight_decay_loss = 0, beta = 0):
    splited_y = tf.split(y, num_splits, axis=1)
    splited_y_ = tf.split(y_, num_splits, axis=1)
    temp_shape = tf.shape(tf.split(splited_y[0], 4, axis=1))

    def one_comb_loss(splited_y, splited_y_, index_list):
        temp = tf.zeros(temp_shape, dtype=tf.float32)
        if use_relative_loss == 1:
            #Relative loss
            for i in range(len(index_list)):
                E, theta, phi, detect = tf.split(splited_y[i], 4, axis=1)
                E_, theta_, phi_, detect_ = tf.split(splited_y_[index_list[i]], 4, axis=1)

                tmp_loss_E = lam_E*tf.square(tf.divide(E-E_, E_+E_offset)*tf.square(detect))
                tmp_loss_theta = lam_theta*tf.square((theta-theta_)*tf.square(detect))
                tmp_loss_phi = lam_phi*tf.square((tf.mod(phi - phi_ + np.pi, 2*np.pi) - np.pi)*tf.square(detect))

                #Normal way of improving the reconstruction of the classification node
                tmp_loss_detect = lam_detect*tf.square(detect-detect_)

                #Adding 100 to the loss for evert gamma-ray that is wrongly paired (just to improve the pairing process)
                tmp_detect_constant = 100*tf.round(tf.abs(tf.subtract(detect, detect_)))

                temp = temp + tmp_loss_E + tmp_loss_theta + tmp_loss_phi + tmp_loss_detect + tmp_detect_constant
        else:
            #Absolute loss
            for i in range(len(index_list)):
                E, theta, phi, detect = tf.split(splited_y[i], 4, axis=1)
                E_, theta_, phi_, detect_ = tf.split(splited_y_[index_list[i]], 4, axis=1)

                tmp_loss_E = lam_E*tf.square((E-E_))#*tf.square(detect))
                tmp_loss_theta = lam_theta*tf.square((theta-theta_))#*tf.square(detect))
                tmp_loss_phi = lam_phi*tf.square((tf.mod(phi - phi_ + np.pi, 2*np.pi) - np.pi))#*tf.square(detect))

                #Normal way of improving the reconstruction of the classification node
                tmp_loss_detect = lam_detect*tf.square(detect-detect_)
                #Adding 100 to the loss for evert gamma-ray that is wrongly paired (just to improve the pairing process)
                tmp_detect_constant = 100*tf.round(tf.abs(tf.subtract(detect, detect_)))

                temp = temp + tmp_loss_E + tmp_loss_theta + tmp_loss_phi + tmp_loss_detect + tmp_detect_constant
        return temp

    # All losses in a list. Inner loop over all permutations.
    list_of_tensors = [one_comb_loss(splited_y, splited_y_, index_list) for index_list in it.permutations(range(num_splits), num_splits)]

    loss = tf.divide(tf.constant(1, dtype=tf.float32), tf.zeros(temp_shape, dtype=tf.float32))
    for i in range(len(list_of_tensors)):
        loss = tf.minimum(loss, list_of_tensors[i])
    return tf.reduce_mean(loss) + beta*weight_decay_loss

'''Similar as network_permutation_sort (relative loss may be implemented) but with classification node
'''
def network_permutation_sort_quattro(y, y_, lam_E = 1, lam_theta = 1, lam_phi = 1, lam_detect = 1, E_offset = 0.1, theta_offset = 0.1, phi_offset = 0.1, use_relative_loss = 0):
    E, theta, phi, E_label, theta_label, phi_label = [], [], [], [], [], []
    sorted_y_ = np.zeros((len(y_),len(y_[0])))

    def one_event_permutation_sort(event_y, event_y_):
        index_min_loss = False
        tmp_min_loss = np.inf
        for index_list in it.permutations(range(int(len(event_y)/4)), int(len(event_y)/4)):
            tmp_loss = 0
            if use_relative_loss == 0:
                #Absolute loss
                for i in range(int(len(event_y)/4)):
                    tmp_detect_constant = 100*np.round(np.abs(event_y[4*i+3] - event_y_[4*index_list[i]+3]))

                    tmp_loss_E = lam_E*np.power((event_y[4*i] - event_y_[4*index_list[i]]), 2)
                    tmp_loss_theta = lam_theta*np.power(event_y[4*i+1] - event_y_[4*index_list[i]+1], 2)
                    tmp_loss_phi = lam_phi*np.power(np.mod(event_y[4*i+2] - event_y_[4*index_list[i]+2] + np.pi, 2*np.pi)-np.pi, 2)
                    tmp_loss_detect = lam_detect*np.power(event_y[4*i+3] - event_y_[4*index_list[i]+3],2)

                    tmp_loss = tmp_loss + tmp_loss_E + tmp_loss_theta + tmp_loss_phi + tmp_loss_detect + tmp_detect_constant
            else:
                #Relative loss
                for i in range(int(len(event_y)/4)):
                    tmp_loss_E = lam_E*np.power((event_y[4*i] - event_y_[4*index_list[i]])/(event_y_[4*index_list[i]] + E_offset), 2)
                    tmp_loss_theta = lam_theta*np.power((event_y[4*i+1] - event_y_[4*index_list[i]+1]), 2)
                    tmp_loss_phi = lam_phi*np.power(np.mod(event_y[4*i+2] - event_y_[4*index_list[i]+2] + np.pi, 2*np.pi)-np.pi, 2)
                    tmp_loss_detect = lam_detect*np.power(event_y[4*i+3] - event_y_[4*index_list[i]+3],2)

                    tmp_loss = tmp_loss + tmp_loss_E + tmp_loss_theta + tmp_loss_phi + tmp_loss_detect
            if tmp_loss < tmp_min_loss:
                tmp_min_loss = tmp_loss
                index_min_loss = index_list
        return index_min_loss

    for i in range(len(y)):
        index_min_loss = one_event_permutation_sort(y[i], y_[i])

        for j in range(int(len(y[0])/4)):
            E.append(y[i][4*j])
            theta.append(y[i][4*j+1])
            phi.append(y[i][4*j+2])
            E_label.append(y_[i][4*index_min_loss[j]])
            theta_label.append(y_[i][4*index_min_loss[j]+1])
            phi_label.append(y_[i][4*index_min_loss[j]+2])

            sorted_y_[i,4*j] = y_[i][4*index_min_loss[j]]
            sorted_y_[i,4*j+1] = y_[i][4*index_min_loss[j]+1]
            sorted_y_[i,4*j+2] = y_[i][4*index_min_loss[j]+2]
            sorted_y_[i,4*j+3] = y_[i][4*index_min_loss[j]+3]

    events = {0: E,
              1: E_label,
              2: theta,
              3: theta_label,
              4: phi,
              5: phi_label}
    return events, sorted_y_

''' Loss function very similar to energy_theta_phi_permutation_loss(). The main difference is that this has a faster permutation sorting using tf.reduce_min() instead of a loop at the end before returning the loss. '''
def energy_theta_phi_permutation_loss_ricky(y, y_, max_multiplicity, lam_E = 1, lam_theta = 1, lam_phi = 1, E_offset = 0.1):
    #Splits y and y_ into {max_multiplicity} different tensors, one for each particle
    splited_y = tf.split(y, max_multiplicity, axis=1)
    splited_y_ = tf.split(y_, max_multiplicity, axis=1)

    #no_events = tf.shape(tf.split(splited_y[0], 3, axis=1))

    #Calculates the loss for one permutation where index_list describes the current permutation
    def one_permutation_loss(splited_y, splited_y_, index_list):
        one_perm_loss = tf.zeros(1, dtype=tf.float32)
        for i in range(len(index_list)):
            #Splits y and y_ into energy and angle data
            E, theta, phi = tf.split(splited_y[i], 3, axis=1)
            E_, theta_, phi_ = tf.split(splited_y_[index_list[i]], 3, axis=1)

            #Actual loss function
            #tmp_loss_E = lam_E*tf.square(tf.divide(E-E_, E_ + E_offset))
            tmp_loss_E = lam_E*(tf.square(E - E_))
            tmp_loss_theta = lam_theta*(tf.square(theta - theta_))
            #tmp_loss_phi = lam_phi*(tf.square(phi - phi_))
            #tmp_loss_theta = lam_theta*tf.square(tf.divide(tf.mod((theta - theta_), np.pi)-np.pi/2, np.pi))
            tmp_loss_phi = lam_phi*tf.square(tf.divide(tf.mod((phi - phi_ + np.pi), 2*np.pi)-np.pi, 2*np.pi)*tf.sin(theta_))
            one_perm_loss = one_perm_loss + tmp_loss_E + tmp_loss_theta + tmp_loss_phi

        return one_perm_loss

    #Lists the possible particle permutations and calculates the losses for each, listed together as output
    possible_permutations = it.permutations(range(max_multiplicity), max_multiplicity)
    all_permutation_losses = [one_permutation_loss(splited_y, splited_y_, index_list) for index_list in possible_permutations]

    # finds the least loss for all events
    min_loss_all_events = tf.reduce_min(all_permutation_losses, axis=0)

    # returns the sum of all minimum losses
    return tf.reduce_sum(min_loss_all_events)
