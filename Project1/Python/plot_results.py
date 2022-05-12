"""Plot results"""

import pickle
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from salamandra_simulation.data import SalamandraData
from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures

from os.path import isfile


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
    grid_id = 100 # change here
    exists = True
    max_iter = 0
    while exists:
        if not isfile('./logs/grid{}/simulation_{}.{}'.format(grid_id,max_iter,'pickle')):
            exists = False
            max_iter = max_iter-1
        else: max_iter = max_iter + 1
        
    results_speed = np.zeros((max_iter+1,3))    
    results_energy = np.zeros((max_iter+1,3))

    
    '''# Load data
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
    amplitudes = parameters.amplitudes
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
    plot_trajectory(head_positions)'''
    for sim_id in range(max_iter+1):    
        data = SalamandraData.from_file('logs/grid{}/simulation_{}.{}'.format(grid_id,sim_id,'h5'))
        with open('logs/grid{}/simulation_{}.pickle'.format(grid_id,sim_id),'rb') as param_file:
            parameters = pickle.load(param_file)
        timestep = data.timestep
        n_iterations = np.shape(data.sensors.links.array)[0]
        times = np.arange(
            start = 0,
            stop = timestep*n_iterations,
            step = timestep,)
        timestep = times[1] - times[0]
        parameters.exercise_8b = False # get rid of this
        if parameters.exercise_8b == True :
            amplitudes = parameters.nominal_amplitudes
            phase_bias = parameters.phase_bias
        if parameters.exercise_8c == True :
            nominal_amplitude_parameters = parameters.nominal_amplitude_parameters
        osc_phases = data.state.phases()
        osc_amplitudes = data.state.amplitudes()
        links_positions = data.sensors.links.urdf_positions()
        head_positions = links_positions[:, 0, :]
        tail_positions = links_positions[:, 8, :]
        joints_positions = data.sensors.joints.positions_all()
        joints_velocities = data.sensors.joints.velocities_all()
        joints_torques = data.sensors.joints.motor_torques_all()
        #     # Notes:
        #     # For the links arrays: positions[iteration, link_id, xyz]
        #     # For the positions arrays: positions[iteration, xyz]
        #     # For the joints arrays: positions[iteration, joint]

        avg_speed = obtain_speed(times,head_positions)
        if parameters.exercise_8b == True :
            results_speed[sim_id,:] = np.hstack((amplitudes[0,0], phase_bias[0,1], avg_speed))
        if parameters.exercise_8c == True:
            results_speed[sim_id,:] = np.hstack((nominal_amplitude_parameters[0], nominal_amplitude_parameters[1], avg_speed))
        tot_energy = np.sum(np.asarray(joints_velocities)*np.asarray(joints_torques)*timestep) 
        
        
        
        results_energy[sim_id,:] = np.hstack((nominal_amplitude_parameters[0], nominal_amplitude_parameters[1], tot_energy))
    
    plt.figure("Speed") 
    plot_2d(results_speed,["R_head","R_tail", "Speed"], n_data=round(np.sqrt(max_iter)))
    plt.figure("Energy")
    plot_2d(results_energy,["R_head","R_tail", "Total energy"], n_data=round(np.sqrt(max_iter)))




    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()

def obtain_speed(times, link_data):
    """Returns the average velocity
    Velocity is calculated as tot_distance/tot_time
    (z is not taken into account)"""
    x = link_data[:,0]
    y = link_data[:,1]

    
    distance = np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2)
    t = times[-1] - times[0]
    return distance/t

def obtain_energy(timestep, joint_velocities, joint_torques):
    """Returns the total energy spent
    It is calculated as the integral of torque*velocity,
    summed for all joints.
    """
    trapz = lambda dt, f_a, f_b : dt * (f_a+f_b)/2
    
    nb_timesteps = np.size(joint_velocities,0)
    nb_joints = np.size(joint_velocities,1)
    total_energy = 0
    
    for j in range(nb_joints):
        for t in range(nb_timesteps-1):
            vel0 = joint_velocities[t,j]
            vel1 = joint_velocities[t+1,j]
            tor0 = joint_torques[t,j]
            tor1 = joint_torques[t+1,j]
            
            total_energy += trapz(timestep, vel0*tor0, vel1*tor1)
    
    return total_energy

if __name__ == '__main__':
    main(plot=not save_plots())
    


