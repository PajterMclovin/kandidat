### Networks running on just Fully Connected (FC) layers.

### FC_energy_theta_phi.py
Network with energy, theta and phi output. Can plot, but is mainly used to used generate a network model (with saved parameters) that is used by FC_plot.py for plotting and evaluation.

### FC_energy_theta_phi_quattro.py
Similar to FC_energy_theta_phi.py but used to implement the classification node. Was created late in the project and is only working for the specific case of data with multiplicities 1-3. The main structure is the same as for FC_energy_theta_phi.py and therefore only the new code with the classification node is commented.

### FC_methods.py
Different methods used by the networks. Note that some of these methods are also used by for example the CNN.

### FC_plot.py
Used to plot and evaluate results from FC_energy_theta_phi.py.

### FC_plot_quattro.py
Used to plot and evaluate results from FC_energy_theta_phi_quattro.py.
