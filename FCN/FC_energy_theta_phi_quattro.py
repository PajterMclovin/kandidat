import tensorflow as tf
import numpy as np
import time
import sys
import importlib, importlib.util
import matplotlib.pyplot as plt

#Rickard's method for importing methods from other directories
def module_from_file(module_name, file_path):
    """
        Used to import modules from different folder in our Github repo.

        module_name is the file name (Ex. 'file.py')
        file_path is the file path (Ex. '../save_load/file.py) if you want to reach a
        folder in parent directory

        Outputs the module as variable that you use as an usual import reference
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

#Importing modules from other directories
import FC_methods as fcm
#fcm = module_from_file("FC_methods", "../../FCN/FC_methods.py") # låt stå
pm = module_from_file('plot_methods.py', '../plot_methods/plot_methods.py')
lf = module_from_file('loss_functions.py', '../loss_functions/loss_functions.py')
sl = module_from_file('ANN_save_load', '../save_load/ANN_save_load.py')

#Adds classification node value to each gamma-ray in labels
def add_label(labels):
    new_labels = np.zeros((len(labels), int(len(labels[0])*4/3)))

    for i in range(int(len(labels[0])/3)):
        new_labels[:,4*i:(4*i+3)] = labels[:,3*i:(3*i+3)]
    
    for i in range(len(labels)):
        for j in range(int(len(labels[0])/3)):
            if labels[i,3*j] != 0:
                new_labels[i,4*j+3] = 1
    return new_labels

#Removes classification node value for each gamma-ray in labels
def remove_label(labels):
    new_labels = np.zeros((len(labels), int(len(labels[0])*3/4)))
    
    for i in range(int(len(new_labels[0])/3)):
        new_labels[:,3*i:(3*i+3)] = labels[:,4*i:(4*i+3)]
        
    return new_labels

def main():
    
    npz_file_name = sys.argv[1]
    no_layers = 10
    no_nodes_per_layer = [128]
    no_iter = 1000
    lam_E = 1
    lam_theta = 1
    lam_phi = 1
    lam_detect = 5                  #(NEW) Weighting factor for the classification node
    E_offset = 0
    theta_offset = 0
    phi_offset = 0
    dropout_rate = 0
    used_data_portion = 1
    beta = 0
    use_relative_loss = 0
    learning_rate = 1e-4
    save_folder_name = 'TEST_NAME'
    
    # print run-arguments to output
    message = 'Test name'
    print('\nnpz_file_name = ', npz_file_name, '\nno_layers = ', no_layers, ', no_nodes_per_layer = ', no_nodes_per_layer, ', no_iter = ', no_iter,',\nlam_E = ', lam_E, ', lam_\u03F4 = ', lam_theta, ', lam_\u03A6 = ', lam_phi, ',\nE_offset = ', E_offset, ', \u03F4_offset = ', theta_offset, ', \u03A6_offset = ', phi_offset, ',\nbeta = ', beta, ', dropout_rate = ', dropout_rate, ', used_data_portion = ', used_data_portion, ',\nsave_folder_name = ', save_folder_name, ', use_relative_loss = ', use_relative_loss,'\n')
    
    #OTHER PARAMETERS
    training_portion = 0.8
    alpha = 0.001
    batch_size = 300
    loss_size = 300
    no_bins = 50
    max_energy_angle_values = [10, np.pi, 2*np.pi]
    
    #Preparing data
    train_data, train_labels, eval_data, eval_labels = fcm.read_data_npz_energy_angle(npz_file_name, training_portion, used_data_portion)
    
    #train_labels, train_data = fcm.add_zero_multiplicity(train_data, train_labels)
    #eval_labels, eval_data = fcm.remove_last_multiplicity(eval_data, eval_labels)
    
    #(NEW) Adds classification labels to the label data sets
    eval_labels = add_label(eval_labels)
    train_labels = add_label(train_labels)
    
    no_output = len(train_labels[0])
    no_input_nodes = len(train_data[0])
    
    #NETWORK STRUCTURE
    #Input and correct labels
    x = tf.placeholder(dtype = tf.float32, shape = [None, no_input_nodes], name = 'x')
    y_ = tf.placeholder(dtype = tf.float32, shape = [None, no_output], name = 'y_')
    
    #y = fcm.hidden_dense_layers(x, no_layers, no_nodes_per_layer, no_output, alpha, dropout_rate)
    y, weight_decay_loss = fcm.hidden_dense_layers_weight_decay(x, no_layers, no_nodes_per_layer, no_output, alpha, dropout_rate, name = 'y')
    
    #(NEW) Using the classification cost function
    loss = lf.energy_theta_phi_permutation_loss_quattro(y, y_, int(no_output/4), lam_E, lam_theta, lam_phi, lam_detect, E_offset, theta_offset, phi_offset, use_relative_loss, weight_decay_loss, beta)
    
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    #Initialize training
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    loss_value_train = []
    loss_value_eval = []
    
    print('\nTraining initialized:')
    init_sec = time.time()
    
    #TRAINING LOOP
    for i in range(0, no_iter):
    
        #Train step with a subset of the training data
        td_sub, tl_sub = fcm.subset(train_data, train_labels, batch_size)
        sess.run(train_step, feed_dict = {x: td_sub, y_: tl_sub})
    
        #Evaluate training every 1 %
        if (i + 1) % (no_iter/100) == 0:
            #Evaluating with data from training set
            td_sub_tmp, tl_sub_tmp = fcm.subset(train_data, train_labels, loss_size)
            loss_value_train_tmp = sess.run(loss, feed_dict={x: td_sub_tmp, y_: tl_sub_tmp})
            loss_value_train.append(loss_value_train_tmp)
    
            #Evaluating with data from evaluation set
            ed_sub_tmp, el_sub_tmp = fcm.subset(eval_data, eval_labels, loss_size)
            loss_value_eval_tmp = sess.run(loss, feed_dict={x: ed_sub_tmp, y_: el_sub_tmp})
            loss_value_eval.append(loss_value_eval_tmp)
    
            #Print above losses every 10 %
            if (i +1) % (no_iter/10) == 0:
                print(np.around((i+1)/no_iter*100,2),'%; #Iter:',i+1, '; Loss (train):', np.around(loss_value_train_tmp,3), '; Loss (eval):', np.around(loss_value_eval_tmp,3))
    
    #Measure training time
    final_sec = time.time()
    delta_sec = final_sec - init_sec
    print('\nTraining time [seconds]:',delta_sec)
    
    # Ricky's save
    list_names = ['loss_value_train', 'loss_value_eval']
    sl.save_results(saver, sess, message ,list_names, loss_value_train, loss_value_eval,
                    save_folder = save_folder_name,
                    iterations=no_iter, training_time=delta_sec, max_multiplicity=int(no_output/3),
                    training_portion = training_portion, used_data_portion = used_data_portion,
                    lam_E = lam_E, lam_theta = lam_theta, lam_phi = lam_phi, lam_detect = lam_detect,
                    E_offset = E_offset, theta_offset = theta_offset, phi_offset = phi_offset,
                    use_relative_loss = use_relative_loss)
    
    #Using the trained network on the whole evaluation set and sorting the output by lowest loss permutation
    y_network = sess.run(y, feed_dict={x: eval_data, y_: eval_labels})
    
    #ONLY WORKS FOR MULTIPLICITY 1-3!
    #(NEW) Calculates the number of correctly reconstructed multiplicities
    correct_multiplicity_count = 0
    for i in range(len(y_network)):
        network_multiplicity_count = 0
        labels_multiplicity_count = 0
        for j in range(int(len(y_network[0])/4)):
            if round(y_network[i, 3+4*j]) == 1:
                network_multiplicity_count += 1
            if eval_labels[i, 3+4*j] == 1:
                labels_multiplicity_count += 1
        if network_multiplicity_count == labels_multiplicity_count:
            correct_multiplicity_count += 1
    print('\nPart_correct_multiplicity = ' + str(np.round(correct_multiplicity_count/len(y_network),3)*100) + ' %')
    
    y_network_with_labels = y_network
    eval_labels_with_labels = eval_labels
    
    #(NEW) Sorting the network output from whole evaluation set
    lam_detect = 100        #Loss factor
    events, eval_labels_with_labels = lf.network_permutation_sort_quattro(y_network, eval_labels, lam_E, lam_theta, lam_phi, lam_detect, E_offset, theta_offset, phi_offset, use_relative_loss)
    #events = fcm.split_energy_angle(y_network, eval_labels)
    events.update({4: np.mod(events[4], 2*np.pi)})
    
    one_count = 0               #Number of real gamma-rays
    correct_total_count = 0     #Number of correctly classified gamma-rays
    correct_zero_count = 0      #Number of correctly classified zero gamma-rays
    correct_one_count = 0       #Number of correctly classified real gamma-rays
    
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
    no_detected_zero_particles = no_particles - one_count
    
    print('Part_correct_detected_particles = ' + str(np.round(correct_total_count/no_particles,3)*100)  + ' %')
    print('Part_correct_detected_zero_particles = ' + str(np.round(correct_zero_count/no_zero_particles,3)*100) + ' %')
    print('Part_correct_detected_one_particles = ' + str(np.round(correct_one_count/(no_particles - no_zero_particles),3)*100) + ' %')
    
    #The following is the same as without the classification node
    error, error_each_bin, mean_std, std_each_bin = pm.error_measurement(events, no_bins, max_energy_angle_values, True)
    
    full_path = './'+save_folder_name
    fig, axs = pm.energy_theta_phi_splitted(events[0], events[2], events[4], events[1], events[3], events[5],
                                 name=save_folder_name, bins = 500, save=True)
    fig.savefig(full_path + '/hist.png')
    plt.close(fig)
    
    # save error measurements
    err_E = 'mean_E = ' + str(np.around(error[0],3))
    err_theta = 'mean_\u03F4 = ' + str(np.around(error[1],3))
    err_phi = 'mean_\u03A6 = ' + str(np.around(error[2],3))
    std_E = 'std_E = ' + str(np.around(mean_std[0],3))
    std_theta = 'std_\u03F4 = ' + str(np.around(mean_std[1],3))
    std_phi = 'std_\u03A6 = ' + str(np.around(mean_std[2],3))
    
    error_file = open(full_path+'/error.txt', 'a+')
    error_file.write('\n' + err_E + '\n' +  err_theta + '\n' +  err_phi + '\n' +  std_E + '\n' +  std_theta + '\n' +  std_phi + '\n')
    error_file.write('\nCorrect detected particles = ' + str(np.round(correct_total_count/no_particles,3)*100)  + ' %\nCorrect detected zero particles = ' + str(np.round(correct_zero_count/no_zero_particles,3)*100) + ' %\nCorrect detected non zero particles = '+ str(np.round(correct_one_count/(no_particles - no_zero_particles),3)*100) + ' %\n')
    error_file.close()
    
    #PLOTTING
    #pm.loss_curves(loss_value_train, loss_value_eval)
    #pm.energy_theta_phi_sum_splitted(events[0], events[2], events[4], events[1], events[3], events[5], int(no_output/3))
    #pm.energy_theta_phi_splitted(events[0], events[2], events[4], events[1], events[3], events[5])
    #pm.error_plot(error_each_bin, std_each_bin, no_bins, max_energy_angle_values, name='')
    
    sess.close()

if __name__ == '__main__':
    main()
