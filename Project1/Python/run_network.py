"""Run network without MuJoCo"""

import time
import numpy as np
import matplotlib.pyplot as plt
from farms_core import pylog
from salamandra_simulation.data import SalamandraState
from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures
from simulation_parameters import SimulationParameters
from network import SalamandraNetwork


def run_network(duration, update=False):
    """Run network without MuJoCo and plot results
    Parameters
    ----------
    duration: <float>
        Duration in [s] for which the network should be run
    update: <bool>
        description
    drive: <float/array>
        Central drive to the oscillators
    """
    # Simulation setup
    timestep = 1e-2
    times = np.arange(0, duration, timestep)
    n_iterations = len(times)
    
    sim_parameters = SimulationParameters()
    state = SalamandraState.salamandra_robotica_2(n_iterations)
    network = SalamandraNetwork(sim_parameters, n_iterations, state)
    
    
    # Logs
    osc_left = np.arange(8)
    osc_right = np.arange(8, 16)
    osc_legs = np.arange(16, 20)
    phases_log = np.zeros([
        n_iterations,
        len(network.state.phases(iteration=0))
    ])
    phases_log[0, :] = network.state.phases(iteration=0)
    amplitudes_log = np.zeros([
        n_iterations,
        len(network.state.amplitudes(iteration=0))
    ])
    amplitudes_log[0, :] = network.state.amplitudes(iteration=0)
    freqs_log = np.zeros([
        n_iterations,
        len(network.robot_parameters.freqs)
    ])
    freqs_log[0, :] = network.robot_parameters.freqs
    #print(freqs_log[0, :])
    outputs_log = np.zeros([
        n_iterations,
        len(network.get_motor_position_output(iteration=0))
    ])
    outputs_log[0, :] = network.get_motor_position_output(iteration=0)
    neural_log = np.zeros([
        n_iterations,
        len(network.outputs(iteration=0))
    ])
    neural_log[0, :] = network.outputs(iteration=0)
    
    # Run network ODE and log data
    drives = np.linspace(0.5,5.5, len(times))
    tic = time.time()
    for i, time0 in enumerate(times[1:]):
        if update:
            network.robot_parameters.update(
                SimulationParameters(drive_mlr = drives[i])            
                )
            
        network.step(i, time0, timestep)
        phases_log[i+1, :] = network.state.phases(iteration=i+1)
        amplitudes_log[i+1, :] = network.state.amplitudes(iteration=i+1)
        outputs_log[i+1, :] = network.get_motor_position_output(iteration=i+1)
        freqs_log[i+1, :] = network.robot_parameters.freqs
        neural_log[i+1, : ] = network.outputs(iteration=i+1)
    toc = time.time()
    
    # Network performance
    pylog.info('Time to run simulation for {} steps: {} [s]'.format(
        n_iterations,
        toc - tic
    ))
    print(neural_log.shape)
    # Implement plots of network results
    thetadot_log = calculate_thetadots(timestep, phases_log)
    #pylog.warning('Implement plots')
    generate_plots(times, phases_log, amplitudes_log, outputs_log, neural_log, freqs_log, thetadot_log, drives)


def calculate_thetadots(timestep, phases_log):
    thetadot_log = np.zeros_like(phases_log)
    for i in range(np.size(phases_log,0)-1):
        #right-side approximation of derivative
        thetadot_log[i,:] = (phases_log[i+1,:] - phases_log[i,:]) / (timestep*2*np.pi)
    return thetadot_log


def generate_plots(times, phases_log, amplitudes_log, outputs_log, neural_log, freqs_log, thetadot_log, drives):
    fig, axs = plt.subplots(4, 1)
    #plot_amplitude(amplitudes_log)
    #plot_output(times, outputs_log, axs)
    plot_neural_output(times, neural_log, axs)
    plot_thetadot(times, thetadot_log, axs)
    plot_drive(times, drives, neural_log, outputs_log, thetadot_log, axs)
    
    plt.show()
    
    
    fig2, axs2 = plt.subplots(2,1)
    plot_freq(drives, freqs_log, axs2)
    plot_amplitude(drives, amplitudes_log, axs2)
    
    plt.show()
    
    fig3, axs3 = plt.subplots(4,1)
    
    # Plot limb and body oscillations
    axs3[0].plot(times, neural_log[:,0],
                 color='k')
    axs3[0].plot(times, neural_log[:,16]-1.5, color='k', linestyle='--')
    axs3[0].set_xticks([])
    axs3[0].set_yticks([])
    axs3[0].set_ylabel('x')
    axs3[0].set_xlim(0, times[-1])
    
    # Plot 
    axs3[1].plot(times, freqs_log[:,1], color='k', label='Body')
    axs3[1].plot(times, freqs_log[:,16], color='k', linestyle='--', label='Limb')
    axs3[1].set_ylim(-0.1,1.5)
    axs3[1].set_xlim(0, times[-1])
    axs3[1].set_xticks([])
    
    axs3[1].set_ylabel('Freq [Hz]')
    
    # Plot amplitude
    axs3[2].plot(times, amplitudes_log[:,1], color='k', label='Body')
    axs3[2].plot(times, amplitudes_log[:,16], color='k', linestyle='--', label='Limb')
    #axs3[2].set_xlim(0,6)
    axs3[2].set_xlim(0, times[-1])
    axs3[2].set_ylim(-0.1,0.7)
    axs3[2].set_ylabel('r')
    axs3[2].set_xticks([])
    
    # Plot freq
    axs3[3].plot(times,drives, 'k')
    axs3[3].hlines([1,3,5],0,times[-1], color='k', linestyle='dotted')
    axs3[3].set_xlabel('Time [s]')
    axs3[3].set_ylabel('d (drive)')
    axs3[3].set_xlim(0, times[-1])
    
    plt.show()



def plot_amplitude(drives, amplitudes_log, axs, third=False):

    axs[1].plot(drives, amplitudes_log[:,1], color='k', label='Body')
    axs[1].plot(drives, amplitudes_log[:,16], color='k', linestyle='--', label='Limb')
    axs[1].set_xlim(0,6)
    axs[1].set_ylim(0,0.7)
    
    axs[1].set_xlabel('drive')
    axs[1].set_ylabel('R')
    
    axs[1].legend()
    
    return
def plot_output(times, outputs_log, axs):
    
    labels_trunk=['x1', 'x2', 'x3', 'x4']
    labels_tail=['x5', 'x6', 'x7', 'x8']
    
    axs[0].plot(times, outputs_log[:,:4] - np.repeat(np.resize(np.linspace(0,4.5*np.pi/3,4),[1,4]),np.size(outputs_log,0),axis=0),
            label=labels_trunk, color='blue')
    axs[0].plot(times, outputs_log[:,4:8] - np.repeat(np.resize(np.linspace(6*np.pi/3,10*np.pi/3,4),[1,4]),np.size(outputs_log,0),axis=0),
            label=labels_tail, color='green')
    axs[0].set_ylabel('x Body')
    axs[0].set_yticklabels([])
    axs[0].set_xlim(0, times[-1])
    #axs[0].legend()

    axs[1].plot(times, outputs_log[:,9], color='blue', label='x17') 
    axs[1].plot(times, outputs_log[:,11]  + 1, color='green', label='x19') 
    axs[1].set_ylabel('x Limb')
    axs[1].set_yticklabels([])
    axs[1].set_xlim(0, times[-1])
    #axs[1].legend()
    
def plot_neural_output(times, outputs_log, axs):
    
    labels_trunk=['x1', 'x2', 'x3', 'x4']
    labels_tail=['x5', 'x6', 'x7', 'x8']
    
    axs[0].plot(times, outputs_log[:,:4] - np.repeat(np.resize(np.linspace(0,4.5*np.pi/3,4),[1,4]),np.size(outputs_log,0),axis=0),
            label=labels_trunk, color='blue')
    axs[0].plot(times, outputs_log[:,4:8] - np.repeat(np.resize(np.linspace(6*np.pi/3,10*np.pi/3,4),[1,4]),np.size(outputs_log,0),axis=0),
            label=labels_tail, color='green')
    axs[0].set_ylabel('x Body')
    axs[0].set_yticklabels([])
    axs[0].set_xticklabels([])
    axs[0].set_xlim(0, times[-1])
    #axs[0].legend()

    axs[1].plot(times, outputs_log[:,16], color='blue', label='x17') 
    axs[1].plot(times, outputs_log[:,18]  - 1.5, color='green', label='x19') 
    axs[1].set_ylabel('x Limb')
    axs[1].set_yticklabels([])
    axs[1].set_ylim([min(outputs_log[:,16])-2, max(outputs_log[:,16])+1.5])
    axs[1].set_xlim(0, times[-1])
    axs[1].set_xticklabels([])
    #axs[1].legend()

def plot_thetadot(times,thetadot_log,axs):
    axs[2].plot(times, thetadot_log, color='k')
    axs[2].set_xlim(0, times[-1])
    axs[2].set_ylabel('Freq [Hz]')
    axs[2].set_ylim(0,1.5*np.max(thetadot_log))
    axs[2].set_xticklabels([])

def plot_freq(drives, freqs_log, axs):

    axs[0].plot(drives, freqs_log[:,1], color='k', label='Body')
    axs[0].plot(drives, freqs_log[:,16], color='k', linestyle='--', label='Limb')
    axs[0].set_xlim(0,6)
    axs[0].set_ylim(0,1.5)
    
    axs[0].set_ylabel('v [Hz]')
    
    axs[0].legend()
    
    

def plot_drive(times, drives, neural_log, outputs_log, thetadot_log, axs):
    
    supLimb= False
    supBody = False
    supAll = False
    for i,d in enumerate(drives):
        if d>1 and not supLimb:
            axs[0].vlines(x=times[i], ymin= -8*np.pi/3, ymax=np.max(neural_log[:,:8])*1.1, linestyle='--', color='grey')
            axs[1].vlines(x=times[i], ymin= -2, ymax=np.max(neural_log[:,16:]+1), linestyle='--', color='grey')
            axs[2].vlines(x=times[i], ymin= 0, ymax=1.5*np.max(thetadot_log), linestyle='--', color='grey')
            axs[3].vlines(x=times[i], ymin= 0, ymax=6, linestyle='--', color='grey')
            axs[3].text(0.2, 0.45,'Walking' ,horizontalalignment='center',
     verticalalignment='center', transform = axs[3].transAxes)
            supLimb = True
        if d>3 and not supBody:
            axs[0].vlines(x=times[i], ymin= -8*np.pi/3, ymax=np.max(neural_log[:,:8]*1.1), linestyle='--', color='grey')
            axs[1].vlines(x=times[i], ymin= -2, ymax=np.max(neural_log[:,16:]+1), linestyle='--', color='grey')
            axs[2].vlines(x=times[i], ymin= 0, ymax=1.5*np.max(thetadot_log), linestyle='--', color='grey')
            axs[3].vlines(x=times[i], ymin= 0, ymax=6, linestyle='--', color='grey')
            axs[3].text(0.6, 0.8,'Swimming' ,horizontalalignment='center',
     verticalalignment='center', transform = axs[3].transAxes)
            supBody = True
        elif d>5 and not supAll:
            axs[0].vlines(x=times[i], ymin= -8*np.pi/3, ymax=np.max(neural_log[:,:8]*1.1), linestyle='--', color='grey')
            axs[1].vlines(x=times[i], ymin= -2, ymax=np.max(neural_log[:,16:]+1), linestyle='--', color='grey')
            axs[2].vlines(x=times[i], ymin= 0, ymax=1.5*np.max(thetadot_log), linestyle='--', color='grey')
            axs[3].vlines(x=times[i], ymin= 0, ymax=6, linestyle='--', color='grey')
            supAll = True
    
    axs[3].plot(times,drives, 'k')
    axs[3].hlines([1,3,5],0,times[-1], color='orange')
    axs[3].set_xlabel('Time [s]')
    axs[3].set_ylabel('drive d')
    axs[3].set_xlim(0, times[-1])
    axs[3].set_ylim(0, 6)
    
    

def main(plot):
    """Main"""

    run_network(duration=40, update=True)

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main(plot=not save_plots())

