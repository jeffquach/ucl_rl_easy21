from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import os

if not os.path.exists('figures'):
    os.makedirs('figures')

# 3D plots taken from this: https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html

def plot_3d(x_axis, y_axis, action_values, file_name):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    dealer_axis = np.arange(1, x_axis, 1)
    player_axis = np.arange(1, y_axis, 1)
    dealer_axis, player_axis = np.meshgrid(dealer_axis, player_axis)
    action_values = np.max(action_values, axis=2).T

    # Plot the surface.
    surf = ax.plot_surface(dealer_axis, player_axis, action_values, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1, 1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig('figures/' + file_name)
    # plt.show()

def line_plot(x_axis, y_axis, x_label, y_label, file_name, legend=None):
    plt.clf()
    for i in range(len(x_axis)):
        plt.plot(x_axis[i], y_axis[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if legend: plt.legend(legend, loc='upper right')
    plt.savefig('figures/' + file_name)
    # plt.show()