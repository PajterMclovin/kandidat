# Importing installed modules
import matplotlib.pyplot as plt
import numpy as np
import sys

# Importing our own modules
import FC_methods as fcm
ANN_save_load = fcm.module_from_file("ANN_save_load", '../save_load/ANN_save_load.py')
pm = fcm.module_from_file("plot_methods","../plot_methods/plot_methods.py")
loss_methods = fcm.module_from_file("loss_functions","../loss_functions/loss_functions.py")
lf = fcm.module_from_file("loss_functions","../loss_functions/loss_functions.py")

#Adds classification node to gamma-rays in labels
def add_label(labels):
    new_labels = np.zeros((len(labels), int(len(labels[0])*4/3)))

    for i in range(int(len(labels[0])/3)):
        new_labels[:,4*i:(4*i+3)] = labels[:,3*i:(3*i+3)]
    
    for i in range(len(labels)):
        for j in range(int(len(labels[0])/3)):
            if labels[i,3*j] != 0:
                new_labels[i,4*j+3] = 1
    return new_labels

#Load network
save_folder = sys.argv[1]
npz_data = sys.argv[2]

print('Loading network from', save_folder)
#list_names = ['loss_value_eval','loss_value_train']
sess, model_vars, x, y, y_ = ANN_save_load.load_model(save_folder)
data, params = ANN_save_load.load_training_results(save_folder)

#Loading data lists
loss_value_train = data['loss_value_train']
loss_value_eval = data['loss_value_eval']

#Loading parameters
multiplicity = int(params['max_multiplicity'])
training_portion = params['training_portion']
used_data_portion = params['used_data_portion']
lam_E = params['lam_E']
lam_theta = params['lam_theta']
lam_phi = params['lam_phi']
E_offset = params['E_offset']
use_relative_loss = params['use_relative_loss']
lam_detect = params['lam_detect']
rel_beta = 0.7

# Get data to use the network on
print("Loading data from", npz_data)
[det_data_train, gun_data_train, det_data_eval, gun_data_eval] = fcm.read_data_npz_energy_angle(npz_data, training_portion, used_data_portion)

#gun_data_eval, det_data_eval = fcm.remove_last_multiplicity(det_data_eval, gun_data_eval, 1)
#gun_data_eval, det_data_eval = fcm.add_zero_multiplicity(det_data_eval, gun_data_eval)
gun_data_eval = add_label(gun_data_eval)
print(np.round(gun_data_eval[0:10,],1))

print('Running network on the data...')
#Reconstruction of the evaluation data set by using the loaded network:
y_network = sess.run(y, feed_dict={x: det_data_eval})
#events = lf.network_permutation_sort(y_network, gun_data_eval, lam_E, lam_theta, lam_phi, E_offset, 0, 0, use_relative_loss)
#events = lf.network_permutation_sort_lab_E_to_beam_E(y_network, gun_data_eval, lam_E, lam_theta, lam_phi, E_offset, 0, 0, use_relative_loss, rel_beta)
#events = fcm.split_energy_angle(y_network, gun_data_eval)
#events, eval_labels_with_labels = lf.network_permutation_sort_quattro(y_network, gun_data_eval, lam_E, lam_theta, lam_phi, 1, E_offset, 0, 0, use_relative_loss)

#events.update({4: np.mod(events[4], 2*np.pi)})
#events[0], events_not_used = fcm.lab_E_to_beam_E(events[0], events[1], events[2], events[3], rel_beta)
#events[0], events[1] = fcm.lab_E_to_beam_E(events[0], events[1], events[2], events[3], rel_beta)

correct_multiplicity_count = 0
for i in range(len(y_network)):
    network_multiplicity_count = 0
    labels_multiplicity_count = 0
    for j in range(int(len(y_network[0])/4)):
        if round(y_network[i, 3+4*j]) == 1:
            network_multiplicity_count += 1
        if gun_data_eval[i, 3+4*j] == 1:
            labels_multiplicity_count += 1
    if network_multiplicity_count == labels_multiplicity_count:
        correct_multiplicity_count += 1
print('\nPart_correct_multiplicity = ' + str(np.round(correct_multiplicity_count/len(y_network),3)*100) + ' %')

y_network_with_labels = y_network
eval_labels_with_labels = gun_data_eval

lam_detect = 100
events, eval_labels_with_labels = lf.network_permutation_sort_quattro(y_network, gun_data_eval, lam_E, lam_theta, lam_phi, lam_detect, E_offset, 0, 0, use_relative_loss)
#events = fcm.split_energy_angle(y_network, eval_labels)
events.update({4: np.mod(events[4], 2*np.pi)})

one_count = 0
correct_total_count = 0
correct_zero_count = 0
correct_one_count = 0

for i in range(len(eval_labels_with_labels)):
    for j in range(int(len(eval_labels_with_labels[0])/4)):
        if round(y_network_with_labels[i, 3+4*j]) == 1:
            one_count += 1            
        if round(y_network_with_labels[i, 3+4*j]) == eval_labels_with_labels[i, 3+4*j]:
            correct_total_count += 1
            if round(y_network_with_labels[i,3+4*j]) == 0:
                correct_zero_count += 1
            else:
                correct_one_count += 1

no_events = len(eval_labels_with_labels)
no_particles = no_events*len(eval_labels_with_labels[0])/4
no_zero_particles = no_particles/3

print('Part_correct_detected_particles = ' + str(np.round(correct_total_count/no_particles,3)*100)  + ' %')
print('Part_correct_detected_zero_perticles = ' + str(np.round(correct_zero_count/no_zero_particles,3)*100) + ' %')
print('Part_correct_detected_one_perticles = ' + str(np.round(correct_one_count/(no_particles - no_zero_particles),3)*100) + ' %')

max_E = np.round(np.amax(events[1]), 1)
no_bins = 50
max_energy_angle_values = [max_E, np.pi, 2*np.pi]

error, error_each_bin, mean_std, std_each_bin = pm.error_measurement(events, no_bins, max_energy_angle_values, True)

#pm.loss_curves(loss_value_train, loss_value_eval)
#pm.energy_theta_phi_sum_splitted(events[0], events[2], events[4], events[1], events[3], events[5], multiplicity)
#pm.energy_theta_phi_splitted(events[0], events[2], events[4], events[1], events[3], events[5])
#pm.error_plot(error_each_bin, std_each_bin, no_bins, max_energy_angle_values, name='')
plt.show()