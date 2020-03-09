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
#fcm = module_from_file("FC_methods", "../../FCN/FC_methods.py")
pm = module_from_file('plot_methods.py', '../plot_methods/plot_methods.py')
lf = module_from_file('loss_functions.py', '../loss_functions/loss_functions.py')
sl = module_from_file('ANN_save_load', '../save_load/ANN_save_load.py')

def main():
    
    #npz_file_name = 'XB_mixed_data_2-3_528339_phi_without_cos.npz'
    
    #IMPORTANT PARAMETERS
    npz_file_name = sys.argv[1]             #String with name of the generated data
    no_layers = 10                          #Number of hidden layers (not including input and output layers)
    no_nodes_per_layer = [128]              #Number of nodes per hidden layer (eg. [128] or [128,256,512])
    no_iter = 10000                         #Number of training iterations
    lam_E = 1                               #Weighting factors in cost function
    lam_theta = 1
    lam_phi = 1
    E_offset = 0                            #Offset value in relative cost function
    dropout_rate = 0                        #Rate of dropout, [0,1] (0 = dropout off)
    used_data_portion = 1.0                 #The used amount data, [0,1]                                
    use_relative_loss = 0                   #0 = NOT USE relative loss function, 1 = USE -||-                    
    save_folder_name = 'TEST_NAME'          #Name of the directory with all the saved data
    
    #Same parameters as above but with command prompt declaration
    '''
    npz_file_name = sys.argv[1]
    no_layers = int(sys.argv[2])
    no_nodes_per_layer = list(map(int, sys.argv[3].split(',')))
    no_iter = int(sys.argv[4])
    lam_E = float(sys.argv[5])
    lam_theta = float(sys.argv[6])
    lam_phi = float(sys.argv[7])
    E_offset = float(sys.argv[8])
    dropout_rate = float(sys.argv[9])
    used_data_portion = float(sys.argv[10])
    use_relative_loss = int(sys.argv[11])
    save_folder_name = 'TEST_NAME'
    '''
    
    #OTHER PARAMETERS
    training_portion = 0.8                  #The portion of the data used for training, [0,1]
    alpha = 0.001                           #Slope for negative values for Leaky ReLu
    batch_size = 300                        #The training batch size
    loss_size = 300                         #Batch size for calculating loss during training
    no_bins = 50                            #Number of bins to calculate standard deviation and error
    beta = 0                                #Weighting factor for the weight decay loss (do not use = 0)
    learning_rate = 1e-4                    #The learning rate
    rel_beta = 0.7                          #The speed of the relativistic beam (only used for relativistic corrections)
    message = 'Test_explanation'            #Text explaning the specific test in saved textfile
    
    #Printing some important parameters
    print('\nnpz_file_name = ', npz_file_name, '\nno_layers = ', no_layers, ', no_nodes_per_layer = ', no_nodes_per_layer, ', no_iter = ', no_iter,',\nlam_E = ', lam_E, ', lam_\u03F4 = ', lam_theta, ', lam_\u03A6 = ', lam_phi, ',\nE_offset = ', E_offset,  ',\nbeta = ', beta, ', dropout_rate = ', dropout_rate, ', used_data_portion = ', used_data_portion, ',\nsave_folder_name = ', save_folder_name, ', use_relative_loss = ', use_relative_loss,'\n')
    
    '''Loading data from the npz file (may not be working when not using phi?)
    train_data = detector data used for training [?x162]
    train_labels = label values corresponding to train_data
    eval_data = detector data used for evaluation [?x162]
    eval_labels = label values corresponding to eval_data'''
    train_data, train_labels, eval_data, eval_labels = fcm.read_data_npz_energy_angle(npz_file_name, training_portion, used_data_portion)
    
    #Methods used when trainging with other multiplicities
    #train_labels, train_data = fcm.add_zero_multiplicity(train_data, train_labels)
    #eval_labels, eval_data = fcm.remove_last_multiplicity(eval_data, eval_labels)
    
    no_output = len(train_labels[0])                    #Width of output layer (max mulitplicity = no_output/3 when using phi)
    no_input_nodes = len(train_data[0])                 #Width of input layer (162)
    max_E = np.round(np.amax(eval_labels),1)            #Maximum correct energy
    max_energy_angle_values = [max_E, np.pi, 2*np.pi]
    
    #NETWORK STRUCTURE
    #Input and correct labels
    x = tf.placeholder(dtype = tf.float32, shape = [None, no_input_nodes], name = 'x')
    y_ = tf.placeholder(dtype = tf.float32, shape = [None, no_output], name = 'y_')
    
    #Hidden layers, using Leaky ReLy and also returning the weight decay loss
    y, weight_decay_loss = fcm.hidden_dense_layers_weight_decay(x, no_layers, no_nodes_per_layer, no_output, alpha, dropout_rate, name = 'y')
    
    #The cost function
    loss = lf.energy_theta_phi_permutation_loss(y, y_, int(no_output/3), lam_E, lam_theta, lam_phi, E_offset, 0, 0, use_relative_loss, weight_decay_loss, beta)
    
    #Setting up training
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    #Storing loss values from training
    loss_value_train = []
    loss_value_eval = []
    
    print('\nTraining initialized:')
    init_sec = time.time()
    
    #TRAINING LOOP
    for i in range(0, no_iter):
    
        #Train step with a subset of the training data
        td_sub, tl_sub = fcm.subset(train_data, train_labels, batch_size)
        sess.run(train_step, feed_dict = {x: td_sub, y_: tl_sub})
    
        #Evaluate training every 1 % of the training iterations
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
    
    #Saving the model, parameters and data
    list_names = ['loss_value_train', 'loss_value_eval']
    sl.save_results(saver, sess, message ,list_names, loss_value_train, loss_value_eval,
                    save_folder = save_folder_name,
                    iterations=no_iter, training_time=delta_sec, max_multiplicity=int(no_output/3),
                    training_portion = training_portion, used_data_portion = used_data_portion,
                    lam_E = lam_E, lam_theta = lam_theta, lam_phi = lam_phi,
                    E_offset = E_offset, theta_offset = 0, phi_offset = 0,
                    use_relative_loss = use_relative_loss)
    
    #Using the trained network on the whole evaluation set
    y_network = sess.run(y, feed_dict={x: eval_data, y_: eval_labels})
    
    #Permutational pairing the gamma-rays to get the lowest loss for y_network (lab_E_to_beam_E = relativistic correction)
    events = lf.network_permutation_sort(y_network, eval_labels, lam_E, lam_theta, lam_phi, E_offset, 0, 0, use_relative_loss)
    #events = lf.network_permutation_sort_lab_E_to_beam_E(y_network, eval_labels, lam_E, lam_theta, lam_phi, E_offset, 0, 0, use_relative_loss, rel_beta)
    
    #Moving all to [0,2*pi]
    events.update({4: np.mod(events[4], 2*np.pi)})
    
    #Relatiistic corrections
    #events[0], events_not_used = fcm.lab_E_to_beam_E(events[0], events[1], events[2], events[3], rel_beta)
    #events[0], events[1] = fcm.lab_E_to_beam_E(events[0], events[1], events[2], events[3], rel_beta)
    
    #Calculating the standard deviations and errors (error & mean_std = total errors & total std)
    error, error_each_bin, mean_std, std_each_bin = pm.error_measurement(events, no_bins, max_energy_angle_values, True)
    
    #PLOTTING
    pm.loss_curves(loss_value_train, loss_value_eval)
    pm.energy_theta_phi_sum_splitted(events[0], events[2], events[4], events[1], events[3], events[5], int(no_output/3))
    pm.energy_theta_phi_splitted(events[0], events[2], events[4], events[1], events[3], events[5])
    pm.error_plot(error_each_bin, std_each_bin, no_bins, max_energy_angle_values, name='')
    
    #Saving plots as png
    full_path = './'+save_folder_name
    fig, axs = pm.energy_theta_phi_splitted(events[0], events[2], events[4], events[1], events[3], events[5],
                                 name=save_folder_name, bins = 500, save=True)
    fig.savefig(full_path + '/hist.png')
    plt.close(fig)
    
    #Save error measurements in textfile
    err_E = 'mean_E = ' + str(np.around(error[0],3))
    err_theta = 'mean_\u03F4 = ' + str(np.around(error[1],3))
    err_phi = 'mean_\u03A6 = ' + str(np.around(error[2],3))
    std_E = 'std_E = ' + str(np.around(mean_std[0],3))
    std_theta = 'std_\u03F4 = ' + str(np.around(mean_std[1],3))
    std_phi = 'std_\u03A6 = ' + str(np.around(mean_std[2],3))
    
    error_file = open(full_path+'/error.txt', 'a+')
    error_file.write('\n' + err_E + '\n' +  err_theta + '\n' +  err_phi + '\n' +  std_E + '\n' +  std_theta + '\n' +  std_phi + '\n')
    error_file.close()

    sess.close()

if __name__ == '__main__':
    main()
