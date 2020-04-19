""" @PETER HALLDESTAM, 2020
    
    Plot methods to analyze the neural network
    
"""
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from itertools import combinations

from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import FancyArrowPatch
from matplotlib.pyplot import close

from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

from utils.help_methods import get_detector_angles

TEXT_SIZE = 17

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot_predictions(y, y_, bins=500, show_detector_angles=False):
    """
    Use to plot a models predictions in similar format as previous years, i.e. 2d histograms ("lasersvärd")
    
    Args:
        prediction : use model.predict(data)
        labels : to compare with predictions
        bins : number of bins i histogram
        permutation : True if network used permutational loss, False if not
        cartesian_coordinate : True if network is trained with cartesin coordinates
        loss_type : used to match permutations
        show_detector_angles : shows points representing the 162 XB detector crystals
        show_description : adds a descriptive title
    Returns:
        figure, axes, events (dict)
    Raises : if data and labels is not the same length
        ValueError : if prediction and labels is not of same length
    """
    if not len(y)==len(y_):
        raise TypeError('The prediction must be of same length as labels.') 
                      
    events = {'predicted_energy': y[::,0::3].flatten(),
              'correct_energy': y_[::,0::3].flatten(), 
              
              'predicted_theta': y[::,1::3].flatten(),
              'correct_theta': y_[::,1::3].flatten(),
              
              'predicted_phi': np.mod(y[::,2::3], 2*np.pi).flatten(),
              'correct_phi': y_[::,2::3].flatten()}
    
    
    fig, axs = plt.subplots(1,3, figsize=(20, 8))
    colormap = truncate_colormap(plt.cm.afmhot, 0.0, 1.0)
    img = []
    img.append(axs[0].hist2d(events['correct_energy'], events['predicted_energy'],cmap=colormap, bins=bins, norm=LogNorm()))
    img.append(axs[1].hist2d(events['correct_theta'], events['predicted_theta'], cmap=colormap, bins=bins, norm=LogNorm()))
    img.append(axs[2].hist2d(events['correct_phi'], events['predicted_phi'], cmap=colormap, bins=bins, norm=LogNorm()))
    
    max_energy = 10
    max_theta = np.pi
    max_phi = 2*np.pi
    line_color = 'blue'
    
    line = np.linspace(0,max_energy)
    for i in range(0,3):
        axs[i].plot(line,line, color=line_color, linewidth = 2, linestyle = '-.')

    if show_detector_angles:
        detector_theta, detector_phi = get_detector_angles()
        axs[1].scatter(detector_theta, detector_theta, marker='x')
        axs[2].scatter(detector_phi, detector_phi, marker='x')
    
    
    axs[0].set_xlabel('Correct E [MeV]', fontsize = TEXT_SIZE)
    axs[1].set_xlabel('Correct \u03F4', fontsize = TEXT_SIZE)
    axs[2].set_xlabel('Correct \u03A6', fontsize = TEXT_SIZE)
    axs[0].set_ylabel('Reconstructed E [MeV]', fontsize = TEXT_SIZE)
    axs[1].set_ylabel('Reconstructed \u03F4', fontsize = TEXT_SIZE)
    axs[2].set_ylabel('Reconstructed E \u03A6', fontsize = TEXT_SIZE)
    
    
    axs[0].set_xlim([0, max_energy])
    axs[0].set_ylim([0, max_energy])
    axs[0].set_aspect('equal', 'box')
    axs[1].set_xlim([0, max_theta])
    axs[1].set_ylim([0, max_theta])
    axs[1].set_aspect('equal', 'box')
    axs[2].set_xlim([0, max_phi])
    axs[2].set_ylim([0, max_phi])
    axs[2].set_aspect('equal', 'box')

    cb1 = fig.colorbar(img[0][3], ax = axs[0], fraction=0.046, pad = 0.04)
    # fig.delaxes(cb1.ax)
    cb2 = fig.colorbar(img[1][3], ax = axs[1], fraction=0.046, pad = 0.04)
    # fig.delaxes(cb2.ax)
    cb3 = fig.colorbar(img[2][3], ax = axs[2], fraction=0.046, pad=0.04)

    cb1.ax.tick_params(labelsize = TEXT_SIZE)
    cb2.ax.tick_params(labelsize = TEXT_SIZE)
    cb3.ax.tick_params(labelsize = TEXT_SIZE)

    axs[0].tick_params(axis='both', which='major', labelsize=TEXT_SIZE)
    axs[1].tick_params(axis='both', which='major', labelsize=TEXT_SIZE)
    axs[2].tick_params(axis='both', which='major', labelsize=TEXT_SIZE)
    
    plt.sca(axs[0])
    plt.xticks(np.linspace(0, 10, 6),['0','2','4','6','8','10'])
    plt.yticks(np.linspace(0, 10, 6),['0','2','4','6','8','10'])
    
    plt.sca(axs[1])
    plt.xticks(np.linspace(0, np.pi, 3),['0','$\pi/2$','$\pi$'])
    plt.yticks(np.linspace(0, np.pi, 3),['0','$\pi/2$','$\pi$'])
    
    plt.sca(axs[2])
    plt.xticks(np.linspace(0, 2*np.pi, 3),['0','$\pi$','$2\pi$'])
    plt.yticks(np.linspace(0, 2*np.pi, 3),['0','$\pi$','$2\pi$'])
    
    fig.tight_layout()
    return fig, events


def plot_loss(history):
    """
    Learning curve; plot training and validation LOSS from model history. Since
    the training loss is obtained from the entire epoch whilst training is it 
    on average measured an 1/2 epoch earlier than the validation loss. To take
    this into account, the training loss curve is shifted half an epoch to the 
    left.
    
    Args:
        history : tf.keras.callbacks.History object returned from model.fit
    Returns:
        -
    Raises:
        TypeError : if history is not History object
    """
    if not isinstance(history, tf.keras.callbacks.History):
        raise TypeError('history must of type tf.keras.callbacks.History')
        
    fig, axs = plt.subplots()
    
    val_loss = history.history['val_loss']
    val_epochs = [i+1 for i in range(len(val_loss))]
    
    train_loss = history.history['loss']
    train_epochs = [i+.5 for i in range(len(train_loss))]
    plt.plot(train_epochs, train_loss, label='training')
    plt.plot(val_epochs, val_loss, label='validation')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    return fig


def plot_energy_distribution(data, bins=100, with_zeros=True):
    """
    Plots the energy distribution in given dataset. Obs: input data i expected
    with the format (energy, theta, phi).
    
    Args:
        data : predicted or label data
        bins : number of bins in histogram
        with_zeros : set False to omit zero energies
    Returns:
        -
    Raises:
        TypeError : if input data is not a numpy array
    """
    if not isinstance(data, np.ndarray):
        raise TypeError('label_data must be an numpy array.')
    
    energy = data[::,0::3].flatten()
    if not with_zeros:
        energy = np.setdiff1d(energy,[0])
    plt.hist(energy, bins, facecolor='blue', alpha=0.5)
    plt.show()
    
    
def plot_depth_mult(json_file):
    """
    Reads json containing mean error from the depth_mult script and returns a
    plot of x=depth, y=mean_error for each max_mult.
    
    """
    fig, axs = plt.subplots(figsize=(10,4))
    
    with open(json_file) as f:
        data = json.load(f)
    
    max_mult = [m for m in list(data.keys())]
    for m in max_mult:
        x, y = [], []
        for depth, error in data[m].items():
            x.append(int(depth))
            y.append(float(error))
        plt.plot(x, y, label=m)
        
    axs.tick_params(axis='both', which='both', labelsize=TEXT_SIZE)
    axs.set_xlabel('depth', fontsize = TEXT_SIZE)
    axs.set_ylabel('mean error [MeV]', fontsize = TEXT_SIZE)
    axs.set_ylim([0, 2])
    axs.yaxis.grid(True, which='Major')
    axs.xaxis.grid(True, which='Major')
    axs.xaxis.set_minor_locator(MultipleLocator(1))
    axs.yaxis.set_minor_locator(MultipleLocator(.1))
    yticks = axs.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.legend(loc='upper left')
    return fig

# test if this shit works
def plot_depth_width(json_file):
    """
    Reads json conatining mean error from the depth_width script and returns a
    contour plot of x=depth, y=width, z=mean_error.
    """
    fig, axs = plt.subplots(figsize=(10,8))
    x, y, z = [], [], []
    
    
    
    with open(json_file) as f:
        data = json.load(f)
    
    widths = [width for width in list(data.keys())]
    for width in widths:
        for depth, mean_error in data[width].items():
            x.append(int(depth))
            y.append(int(width))
            z.append(float(mean_error))
            
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    
    xi = np.linspace(0, np.max(x), 100)
    yi = np.linspace(0, np.max(y), 100)
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')

    plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
    axs.contourf(xi, yi, zi, 15, cmap=plt.cm.jet)
    plt.scatter(x, y, marker='x', c='k', s=20)
    
    cbar = plt.colorbar(ax=axs)
    cbar.ax.tick_params(labelsize=TEXT_SIZE)    
    axs.tick_params(axis='both', which='both', labelsize=TEXT_SIZE)
    axs.set_xlabel('depth', fontsize=TEXT_SIZE)
    axs.set_ylabel('width', fontsize=TEXT_SIZE)
    cbar.set_label('mean error [MeV]', fontsize=TEXT_SIZE)
    return fig



##-------------------- experimental shit --------------------------------------
def plot_sphere_diagram(y, y_, event=0):
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    R = 2
    
    
    # Make data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    plt.axis('off')
    ax.set_proj_type('persp')
    
    # Plot the surface
    ax.plot_surface(sx, sy, sz, color='b', alpha=.1)
    
    
    # ax.scatter([0],[0],[0],color="g",s=100)
    
    #plot axis
    bx = Arrow3D([-2,2],[0,0],[0,0], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    by = Arrow3D([0,0],[2,-2],[0,0], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    bz = Arrow3D([0,0],[0,0],[-2,2], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    ax.add_artist(bx)
    ax.add_artist(by)
    ax.add_artist(bz)

    #plot correct
    px = y_[event,0::3]/10
    py = y_[event,1::3]/10
    pz = y_[event,2::3]/10
    for i in range(len(px)):
        print([-px[i],0], [py[i],0], [-pz[i],0])
        p0 = Arrow3D([0, 5*px[i]], [0, 5*py[i]], [0, 5*pz[i]], lw=1, color="k")
        p = Arrow3D([0, px[i]], [0, py[i]], [0, pz[i]],
                    mutation_scale=10,lw=2, arrowstyle="-|>", color="r")
        ax.add_artist(p0)
        ax.add_artist(p)
    
    #plot predicted
    px = y[event,0::3]/10
    py = y[event,1::3]/10
    pz = y[event,2::3]/10
    for i in range(len(px)):
        print([-px[i],0], [py[i],0], [-pz[i],0])
        p0 = Arrow3D([0, 5*px[i]], [0, 5*py[i]], [0, 5*pz[i]], lw=1, color="k")
        p = Arrow3D([0, px[i]], [0, py[i]], [0, pz[i]],
                    mutation_scale=10,lw=2, arrowstyle="-|>", color="b")
        
        ax.add_artist(p0)
        ax.add_artist(p)
            
    ax.view_init(40,-45)
    plt.show()

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)



