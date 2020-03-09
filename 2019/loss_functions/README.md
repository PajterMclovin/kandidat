### cost_functions.py
Containing the 3 different cost functions used, all based on the first one; energy_theta_phi_permutation_loss

energy_theta_phi_permutation_loss  - is the standard cost function, using the permutational pairing process. Can be used for both relativistic and non relativistic data. Has both the absolute and relative cost functions implemented.

energy_theta_phi_permutation_loss_lab_E_to_beam_E  - is a version of the standard cost function that applies the Doppler-correction in the cost function; training the network to be optimized to reconstruct the ENERGY in the beam frame. Observe that these networks still outputs the energy (and angles) in the lab frame. Do not have the relative cost function implemented.

energy_theta_phi_permutation_loss_quattro - is a version of the standard cost function with the classification node implemented. Seems to have the relative cost function implemented, but I am not sure.

To each of the 3 cost functions there is a corresponding sorting method: network_permutation_sort_..., that is used pair the gamma-rays in the data sets returned when the networks are run on evaluation data. These sorting methods are created to do the exact same pairing as in the cost functions.
