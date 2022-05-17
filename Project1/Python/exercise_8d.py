"""Exercise 8d"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
import time
import matplotlib.pyplot as plt
from farms_core import pylog
from salamandra_simulation.data import SalamandraState
from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures
from network import SalamandraNetwork

def exercise_8d1(timestep, duration=20, update=True):
    """Exercise 8d1"""
    drive_params = {
        "baseline": 3.5,
        "delta_turn": 1,
        "turn_start": 2,
        "turn_duration": 5,
        "direction": "right"}
    times = np.arange(0, duration, timestep)
    n_iterations = len(times)
    
    drive_mlr  = drive_params["baseline"]*np.ones(len(times))
    drive_offset_turn = drive_params["delta_turn"]
    
    start     = drive_params["turn_start"]
    duration  = drive_params["turn_duration"]
    
    turns = ["None" for i in range(n_iterations)]
    for i, t in enumerate(times):
        if t>=start and t<=start + duration:
            turns[i] = drive_params["direction"]
            
        
    sim_parameters = SimulationParameters(
        drive_mlr = drive_mlr[0],
        drive_offset_turn = drive_offset_turn,
    )        
    
    state, network, logs = init_network(n_iterations, sim_parameters)
    
    

    
    tic = time.time()
    for i, time0 in enumerate(times[1:]):
        if update:
            network.robot_parameters.update(
                SimulationParameters(
                    drive_mlr = drive_mlr[i],
                    turn = turns[i]
                )
            )
        network.step(i, time0, timestep)
        
        logs = update_logs(logs, i, network)
    
    logs["thetadot_log"] = calculate_thetadots(timestep, logs["phases_log"])
    # # Alternative option
    # phases_log[:, :] = network.state.phases()
    # amplitudes_log[:, :] = network.state.amplitudes()
    # outputs_log[:, :] = network.get_motor_position_output()
    toc = time.time()

    # Network performance
    pylog.info('Time to run simulation for {} steps: {} [s]'.format(
        n_iterations,
        toc - tic
    ))

    # Implement plots of network results
    pylog.warning('Implement plots')
    
    os.makedirs('./logs/8d/', exist_ok=True)
    filename = './logs/grid/simulation.{}'
        
    # Log simulation parameters
    with open(filename.format('pickle'), 'wb') as param_file:
        pickle.dump(sim_parameters, param_file)
        pickle.dump(logs,param_file)
    
    #generate_plots(times, phases_log, amplitudes_log, outputs_log, freqs_log, thetadot_log, drives_left_log, drives_right_log)

# =============================================================================
# Debugging with GUI
#     parameter_set = [
#         SimulationParameters(
#             duration=10,  # Simulation duration in [s]
#             timestep=timestep,  # Simulation timestep in [s]
#             spawn_position=[0, 0, 0.1],  # Robot position in [m]
#             spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
#             drive_mlr=drive_mlr[0],  # An example of parameter part of the grid search
#             turn=drive_params["direction"],  # Another example
#             # ...
#         )
#         #for drive in np.linspace(3, 4, 2)
#         # for amplitudes in ...
#         # for ...
#     ]
#     os.makedirs('./logs/example/', exist_ok=True)
#     for simulation_i, sim_parameters in enumerate(parameter_set):
#         filename = './logs/example/simulation_{}.{}'
#         sim, data = simulation(
#             sim_parameters=sim_parameters,  # Simulation parameters, see above
#             arena='water',  # Can also be 'ground', give it a try!
#             # fast=True,  # For fast mode (not real-time)
#             # headless=True,  # For headless mode (No GUI, could be faster)
#             # record=True,  # Record video
#         )
# =============================================================================

def exercise_8d2(timestep):
    """Exercise 8d2"""
    # Use exercise_example.py for reference
    pass


def init_network(n_iterations, sim_parameters):    
    state = SalamandraState.salamandra_robotica_2(n_iterations)
    network = SalamandraNetwork(sim_parameters, n_iterations, state)

    logs = dict()
    # Logs
    logs["phases_log"] = np.zeros([
        n_iterations,
        len(network.state.phases(iteration=0))
    ])
    logs["phases_log"][0, :] = network.state.phases(iteration=0)
    amplitudes_log = np.zeros([
        n_iterations,
        len(network.state.amplitudes(iteration=0))
    ])
    logs["amplitudes_log"][0, :] = network.state.amplitudes(iteration=0)
    freqs_log = np.zeros([
        n_iterations,
        len(network.robot_parameters.freqs)
    ])
    logs["freqs_log"][0, :] = network.robot_parameters.freqs
    logs["outputs_log"] = np.zeros([
        n_iterations,
        len(network.get_motor_position_output(iteration=0))
    ])
    logs["d_l_log"] = np.zeros([
        n_iterations,
        1
    ])
    logs["d_l_log"][0,:] = network.robot_parameters.d_l
    logs["d_r_log"] = np.zeros([
        n_iterations,
        1
    ])
    logs["d_r_log"][0,:] = network.robot_parameters.d_r
    
    logs["output_log"][0, :] = network.get_motor_position_output(iteration=0)
    
    return state, network, logs

def update_logs(logs, i, network):
    logs["phases_log"][i+1, :] = network.state.phases(iteration=i+1)
    logs["amplitudes_log"][i+1, :] = network.state.amplitudes(iteration=i+1)
    logs["outputs_log"][i+1, :] = network.get_motor_position_output(iteration=i+1)
    logs["freqs_log"][i+1, :] = network.robot_parameters.freqs
    logs["drives_left_log"][i+1, :] = network.robot_parameters.d_l
    logs["drives_right_log"][i+1, :] = network.robot_parameters.d_r

def calculate_thetadots(timestep, phases_log):
    thetadot_log = np.zeros_like(phases_log)
    for i in range(np.size(phases_log,0)-1):
        #right approximation of derivative
        thetadot_log[i,:] = (phases_log[i+1,:] - phases_log[i,:]) / (timestep*2*np.pi)
    return thetadot_log


    

def generate_plots(times, phases_log, amplitudes_log, outputs_log, freqs_log, thetadot_log, drives_left_log, drives_right_log):
    fig, axs = plt.subplots(4, 1)
    #plot_amplitude(amplitudes_log)
    plot_output(times, outputs_log, axs)
    plot_thetadot(times, thetadot_log, axs)
    plot_drive(times, drives_left_log,  outputs_log, thetadot_log, axs)
    plot_drive(times, drives_right_log, outputs_log, thetadot_log, axs)
    
    plt.show()
    
    
    fig2, axs2 = plt.subplots(2,1)
    plot_freq(drives_left_log, freqs_log, axs2)
    plot_amplitude(drives_left_log, amplitudes_log, axs2)
    plot_freq(drives_right_log, freqs_log, axs2)
    plot_amplitude(drives_right_log, amplitudes_log, axs2)
    
    
    plt.show()


def plot_amplitude(drives, amplitudes_log, axs):
    axs[1].plot(drives, amplitudes_log[:,1], color='k', label='Body')
    axs[1].plot(drives, amplitudes_log[:,16], color='k', linestyle='--', label='Limb')
    axs[1].set_xlim(0,6)
    axs[1].set_ylim(0,0.7)
    
    axs[1].set_xlabel('drive')
    axs[1].set_ylabel('R')
    
    axs[1].legend()
    
    return
def plot_output(times, outputs_log, axs):
    print(outputs_log.shape)
    axs[0].plot(times, outputs_log[:,:8] - np.repeat(np.resize(np.linspace(0,8*np.pi/3,8),[1,8]),np.size(outputs_log,0),axis=0))
    axs[0].set_ylabel('x Body')
    axs[0].set_yticklabels([])
    axs[0].set_xlim(0, times[-1])
    
    axs[1].plot(times, outputs_log[:,9], color='blue')
    axs[1].plot(times, outputs_log[:,11]  + 1, color='green')
    axs[1].set_ylabel('x Limb')
    axs[1].set_yticklabels([])
    axs[1].set_xlim(0, times[-1])

def plot_thetadot(times,thetadot_log,axs):
    axs[2].plot(times, thetadot_log, color='k')
    axs[2].set_xlim(0, times[-1])
    axs[2].set_ylabel('Freq [Hz]')

def plot_freq(drives, freqs_log, axs):

    axs[0].plot(drives, freqs_log[:,1], color='k', label='Body')
    axs[0].plot(drives, freqs_log[:,16], color='k', linestyle='--', label='Limb')
    axs[0].set_xlim(0,6)
    axs[0].set_ylim(0,1.5)
    
    axs[0].set_ylabel('v [Hz]')
    
    axs[0].legend()
    
    

def plot_drive(times, drives, outputs_log, thetadot_log, axs):
    
    supLimb= False
    supBody = False
    supAll = False
    for i,d in enumerate(drives):
        if d>1 and not supLimb:
            axs[0].vlines(x=times[i], ymin= -8*np.pi/3, ymax=np.max(outputs_log[:,:8])*1.1, linestyle='--', color='grey')
            axs[1].vlines(x=times[i], ymin= 0, ymax=np.max(outputs_log[:,8:]+1), linestyle='--', color='grey')
            axs[2].vlines(x=times[i], ymin= 0, ymax=np.max(thetadot_log), linestyle='--', color='grey')
            axs[3].vlines(x=times[i], ymin= 0, ymax=6, linestyle='--', color='grey')
            axs[3].text(0.2, 0.45,'Walking' ,horizontalalignment='center',
     verticalalignment='center', transform = axs[3].transAxes)
            supLimb = True
        if d>3 and not supBody:
            axs[0].vlines(x=times[i], ymin= -8*np.pi/3, ymax=np.max(outputs_log[:,:8]*1.1), linestyle='--', color='grey')
            axs[1].vlines(x=times[i], ymin= 0, ymax=np.max(outputs_log[:,8:]+1), linestyle='--', color='grey')
            axs[2].vlines(x=times[i], ymin= 0, ymax=np.max(thetadot_log), linestyle='--', color='grey')
            axs[3].vlines(x=times[i], ymin= 0, ymax=6, linestyle='--', color='grey')
            axs[3].text(0.6, 0.8,'Swimming' ,horizontalalignment='center',
     verticalalignment='center', transform = axs[3].transAxes)
            supBody = True
        elif d>5 and not supAll:
            axs[0].vlines(x=times[i], ymin= -8*np.pi/3, ymax=np.max(outputs_log[:,:8]*1.1), linestyle='--', color='grey')
            axs[1].vlines(x=times[i], ymin= 0, ymax=np.max(outputs_log[:,8:]+1), linestyle='--', color='grey')
            axs[2].vlines(x=times[i], ymin= 0, ymax=np.max(thetadot_log), linestyle='--', color='grey')
            axs[3].vlines(x=times[i], ymin= 0, ymax=6, linestyle='--', color='grey')
            supAll = True
    
    axs[3].plot(times,drives, 'k')
    axs[3].hlines([1,3,5],0,times[-1], color='orange')
    axs[3].set_xlabel('Time [s]')
    axs[3].set_ylabel('drive d')
    axs[3].set_xlim(0, times[-1])
    axs[3].set_ylim(0, 6)

if __name__ == '__main__':
    exercise_8d1(timestep=1e-2)
    exercise_8d2(timestep=1e-2)

