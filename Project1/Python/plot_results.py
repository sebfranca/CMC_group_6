"""Plot results"""

import pickle
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from salamandra_simulation.data import SalamandraData
from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures


def plot_positions(times, link_data):
    """Plot positions"""
    for i, data in enumerate(link_data.T):
        plt.plot(times, data, label=['x', 'y', 'z'][i])
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [m]')
    plt.grid(True)


def plot_trajectory(link_data):
    """Plot positions"""
    plt.plot(link_data[:, 0], link_data[:, 1])
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.grid(True)


def plot_1d(results, labels):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 2].

    labels - The labels should be a list of two string for the xlabel and the
    ylabel (in that order).
    """

    plt.plot(results[:, 0], results[:, 1], marker='.')

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.grid(True)


def plot_2d(results, labels, n_data=300, log=False, cmap=None):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 3].

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    n_data - Represents the number of points used along x and y to draw the plot

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    xnew = np.linspace(min(results[:, 0]), max(results[:, 0]), n_data)
    ynew = np.linspace(min(results[:, 1]), max(results[:, 1]), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)
    results_interp = griddata(
        (results[:, 0], results[:, 1]), results[:, 2],
        (grid_x, grid_y),
        method='linear',  # nearest, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    plt.plot(results[:, 0], results[:, 1], 'r.')
    imgplot = plt.imshow(
        results_interp,
        extent=extent,
        aspect='auto',
        origin='lower',
        interpolation='none',
        norm=LogNorm() if log else None
    )
    if cmap is not None:
        imgplot.set_cmap(cmap)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    cbar = plt.colorbar()
    cbar.set_label(labels[2])


def main(plot=True):
    """Main"""
    # Load data
    data = SalamandraData.from_file('logs/example/simulation_0.h5')
    with open('logs/example/simulation_0.pickle', 'rb') as param_file:
        parameters = pickle.load(param_file)
    timestep = data.timestep
    n_iterations = np.shape(data.sensors.links.array)[0]
    times = np.arange(
        start=0,
        stop=timestep*n_iterations,
        step=timestep,
    )
    timestep = times[1] - times[0]
    amplitudes = parameters.nominal_amplitudes
    phase_lag_body = parameters.phase_lag_body
    osc_phases = data.state.phases()
    osc_amplitudes = data.state.amplitudes()
    links_positions = data.sensors.links.urdf_positions()
    head_positions = links_positions[:, 0, :]
    tail_positions = links_positions[:, 8, :]
    joints_positions = data.sensors.joints.positions_all()
    joints_velocities = data.sensors.joints.velocities_all()
    joints_torques = data.sensors.joints.motor_torques_all()
    # Notes:
    # For the links arrays: positions[iteration, link_id, xyz]
    # For the positions arrays: positions[iteration, xyz]
    # For the joints arrays: positions[iteration, joint]

    # Plot data
    plt.figure('Positions')
    plot_positions(times, head_positions)
    plt.figure('Trajectory')
    plot_trajectory(head_positions)

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main(plot=not save_plots())

