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


def plot_trajectory(link_data, axs=None, subidx = 0, label = None):
    """Plot positions"""
    if not isinstance(subidx,int):
        plt.plot(link_data[:, 0], link_data[:, 1])
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')
        plt.grid(True)
    else:
        axs[subidx].plot(link_data[:, 0], link_data[:, 1], label = label)
        axs[subidx].set_xlabel('x [m]')
        axs[subidx].set_ylabel('y [m]')
        axs[subidx].axis('equal')
        axs[subidx].legend()
        axs[subidx].grid(True)


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

            
def obtain_speed(times, link_data):
    """Returns the average velocity
    Velocity is calculated as tot_distance/tot_time
    (z is not taken into account)
    
    We start from index 1000 (t=10s) to avoid accounting for transient
    """
    x = link_data[:,0]
    y = link_data[:,1]
        
    distance = np.sqrt((x[-1]-x[1000])**2 + (y[-1]-y[1000])**2)
    t = times[-1] - times[0]
    return distance/t


def main(plot=True, ex_id = '8b', grid_id = 0):
    """Main"""
    if ex_id == '8b' or ex_id == '8c':
        exists=True
        max_iter = 0
        while exists:
            if not isfile('./logs/ex_{}/grid{}/simulation_{}.{}'.format(ex_id,grid_id,max_iter,'pickle')):
                exists = False
                max_iter = max_iter-1
            else: max_iter = max_iter + 1
        
        results_speed = np.zeros((max_iter+1,3))    
        results_energy = np.zeros((max_iter+1,3))
        results_objective = np.zeros((max_iter+1,3))
        
        for sim_id in range(max_iter+1):    
            #Load data
            data = SalamandraData.from_file('logs/ex_{}/grid{}/simulation_{}.{}'.format(ex_id,grid_id,sim_id,'h5'))
            with open('logs/ex_{}/grid{}/simulation_{}.pickle'.format(ex_id,grid_id,sim_id),'rb') as param_file:
                parameters = pickle.load(param_file)
            timestep = data.timestep
            n_iterations = np.shape(data.sensors.links.array)[0]
            times = np.arange(
                start=0,
                stop = timestep*n_iterations,
                step = timestep,)
            timestep = times[1] - times[0]
            if parameters.exercise_8b == True:
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
                results_speed[sim_id,:] = np.hstack((amplitudes[0], phase_bias[0,1]/(2*np.pi/8), avg_speed))
            if parameters.exercise_8c == True:
                results_speed[sim_id,:] = np.hstack((nominal_amplitude_parameters[0], nominal_amplitude_parameters[1], avg_speed))
            
            #Total energy = Sum(velocity*torque*timestep)
            # We start from index 1000 (t=10s) to remove transient
            tot_energy=np.sum(np.abs(np.asarray(joints_velocities[1000:,:])*np.asarray(joints_torques[1000:,:])*timestep))
            #objective: kinetic energy per unit mass / energy to move the joints
            #do not calculate it if the solution is really bad
            if avg_speed > 0.01 and tot_energy > 1:
                objective = avg_speed**2 / tot_energy
            else:
                objective = 0
            
            if parameters.exercise_8b == True :
                results_energy[sim_id,:] =  np.hstack((amplitudes[0], phase_bias[0,1]/(2*np.pi/8), tot_energy))
                results_objective[sim_id,:] = np.hstack((amplitudes[0], phase_bias[0,1]/(2*np.pi/8), objective))
            if parameters.exercise_8c == True :
                results_energy[sim_id,:] = np.hstack((nominal_amplitude_parameters[0], nominal_amplitude_parameters[1], tot_energy))
        
        
        if parameters.exercise_8b == True :
            plt.figure("Speed") 
            plot_2d(results_speed,["Body amplitude [arbitrary units]","Phase lag [nb of S shapes]", "Average speed [m/s]"], n_data=round(np.sqrt(max_iter)))
            plt.savefig("8b_speed.png")
            plt.figure("Energy")
            plt.savefig("8b_energy.png")
            plot_2d(results_energy,["Body amplitude [arbitrary units]","Phase lag [nb of S shapes]", "Total energy [J]"], n_data=round(np.sqrt(max_iter)))
            plt.figure("Speed/energy")
            plot_2d(results_objective,["Body amplitude [arbitrary units]","Phase lag [nb of S shapes]", "Speed^2/Energy [kg^-1]"], n_data=round(np.sqrt(max_iter)))
            plt.savefig("8b_objective.png")
            
        if parameters.exercise_8c == True :
            plt.figure("Speed") 
            plot_2d(results_speed,["R_head","R_tail", "Speed"], n_data=round(np.sqrt(max_iter)))
            plt.figure("Energy")
            plot_2d(results_energy,["R_head","R_tail", "Total energy"], n_data=round(np.sqrt(max_iter)))
    
    
    
    elif ex_id in ['8d1','8d2']:
        #Load data
        data = SalamandraData.from_file('logs/ex_{}/simulation.{}'.format(ex_id,'h5'))
        with open('logs/ex_{}/simulation.pickle'.format(ex_id),'rb') as param_file:
            parameters = pickle.load(param_file)
        timestep = data.timestep
        n_iterations = np.shape(data.sensors.links.array)[0]
        times = np.arange(
            start=0,
            stop = timestep*n_iterations,
            step = timestep,)
        timestep = times[1] - times[0]
        
        osc_phases = data.state.phases()
        osc_amplitudes = data.state.amplitudes()
        links_positions = data.sensors.links.urdf_positions()
        head_positions = links_positions[:, 0, :]
        tail_positions = links_positions[:, 8, :]
        joints_positions = data.sensors.joints.positions_all()
        joints_velocities = data.sensors.joints.velocities_all()
        joints_torques = data.sensors.joints.motor_torques_all()
        
        fig, axs = plt.subplots(2,1)
        axs[1].scatter(0,0, c='k',label = "Initial position")
        plot_trajectory(head_positions,axs=axs,subidx=1, label="Head trajectory")
        plot_trajectory(tail_positions,axs=axs,subidx=1, label="Tail trajectory")
              
        for j in range(8):
            axs[0].plot(times,np.asarray(joints_positions[:,j])-j, label = "Joint "+str(j))
        #axs[1].legend(loc='best')
        axs[0].set_xlabel("Time(s)")
        axs[0].set_yticklabels([])
        axs[0].set_ylabel("Joint positions")
        plt.savefig("{}.png".format(ex_id))
        plt.show()
        
        
        
        
        
    elif ex_id in ['8e1','8e2']:
        #Load data
        data = SalamandraData.from_file('logs/ex_{}/simulation.{}'.format(ex_id,'h5'))
        with open('logs/ex_{}/simulation.pickle'.format(ex_id),'rb') as param_file:
            parameters = pickle.load(param_file)
        timestep = data.timestep
        n_iterations = np.shape(data.sensors.links.array)[0]
        times = np.arange(
            start=0,
            stop = timestep*n_iterations,
            step = timestep,)
        timestep = times[1] - times[0]
        
        osc_phases = data.state.phases()
        osc_amplitudes = data.state.amplitudes()
        links_positions = data.sensors.links.urdf_positions()
        head_positions = links_positions[:, 0, :]
        tail_positions = links_positions[:, 8, :]
        joints_positions = data.sensors.joints.positions_all()
        joints_velocities = data.sensors.joints.velocities_all()
        joints_torques = data.sensors.joints.motor_torques_all()
        
        fig, axs = plt.subplots(2,1)
        axs[1].scatter(0,0, c='k',label = "Initial position")
        plot_trajectory(head_positions,axs=axs,subidx=1, label="Head trajectory")
              
        for j in range(8):
            axs[0].plot(times,np.cos(osc_phases[:,j])-3*j, label = "Joint "+str(j))
        
        axs[0].set_xlabel("Time(s)")
        axs[0].set_yticklabels([])
        axs[0].set_ylabel("Phase")
        
        avg_speed = obtain_speed(times,head_positions)
        tot_energy=np.sum(np.abs(np.asarray(joints_velocities[1000:,:])*np.asarray(joints_torques[1000:,:])*timestep))
        if avg_speed > 0.01 and tot_energy > 1:
            objective = avg_speed**2 / tot_energy
        else:
            objective = 0
            
        print("Average speed : {} \nTotal energy: {} \nObjective function: {}".format(
            avg_speed,tot_energy,objective))
        


if __name__ == '__main__':
    main(plot=not save_plots(), ex_id = '8e2', grid_id='_final2')

