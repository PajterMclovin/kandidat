# Importing installed modules
import matplotlib.pyplot as plt
import numpy as np
import sys

#Importing our own modules
import FC_methods as fcm
ANN_save_load = fcm.module_from_file("ANN_save_load", '../save_load/ANN_save_load.py')
pm = fcm.module_from_file("plot_methods","../plot_methods/plot_methods.py")
loss_methods = fcm.module_from_file("loss_functions","../loss_functions/loss_functions.py")
lf = fcm.module_from_file("loss_functions","../loss_functions/loss_functions.py")

#Specify what model/data to import
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
rel_beta = 0.7

print("Loading data from", npz_data)
[det_data_train, gun_data_train, det_data_eval, gun_data_eval] = fcm.read_data_npz_energy_angle(npz_data, training_portion, used_data_portion)

#gun_data_eval, det_data_eval = fcm.add_zero_multiplicity(det_data_eval, gun_data_eval)
#gun_data_eval, det_data_eval = fcm.remove_last_multiplicity(det_data_eval, gun_data_eval)

print('Running network on the data...')
#Reconstruction of the evaluation data set by using the loaded network:
y_network = sess.run(y, feed_dict={x: det_data_eval})
events = lf.network_permutation_sort(y_network, gun_data_eval, lam_E, lam_theta, lam_phi, E_offset, 0, 0, use_relative_loss)
#events = lf.network_permutation_sort_lab_E_to_beam_E(y_network, gun_data_eval, lam_E, lam_theta, lam_phi, E_offset, 0, 0, use_relative_loss, rel_beta)
#events = fcm.split_energy_angle(y_network, gun_data_eval)

events.update({4: np.mod(events[4], 2*np.pi)})
#events[0], events_not_used = fcm.lab_E_to_beam_E(events[0], events[1], events[2], events[3], rel_beta)
#events[0], events[1] = fcm.lab_E_to_beam_E(events[0], events[1], events[2], events[3], rel_beta)

max_E = np.round(np.amax(events[1]), 1)
no_bins = 50
max_energy_angle_values = [max_E, np.pi, 2*np.pi]

#error, error_each_bin, mean_std, std_each_bin = pm.error_measurement(events, no_bins, max_energy_angle_values, True)

#pm.loss_curves(loss_value_train, loss_value_eval)
#pm.energy_theta_phi_sum_splitted(events[0], events[2], events[4], events[1], events[3], events[5], multiplicity)
pm.energy_theta_phi_splitted(events[0], events[2], events[4], events[1], events[3], events[5])
#pm.error_plot(error_each_bin, std_each_bin, no_bins, max_energy_angle_values, name='')
plt.show()