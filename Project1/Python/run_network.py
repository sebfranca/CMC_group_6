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


def run_network(duration, update=False, drive=0):
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
    sim_parameters = SimulationParameters(
        drive_mlr=drive,
        amplitude_gradient=None,
        phase_lag_body=None,
        turn=None,
    )
    state = SalamandraState.salamandra_robotica_2(n_iterations)
    network = SalamandraNetwork(sim_parameters, n_iterations, state)
    osc_left = np.arange(8)
    osc_right = np.arange(8, 16)
    osc_legs = np.arange(16, 20)

    # Logs
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
    outputs_log = np.zeros([
        n_iterations,
        len(network.get_motor_position_output(iteration=0))
    ])
    outputs_log[0, :] = network.get_motor_position_output(iteration=0)
    
    drives = np.linspace(0.5,5.5, len(times))

    # Run network ODE and log data
    tic = time.time()
    for i, time0 in enumerate(times[1:]):
        if update:
            network.robot_parameters.update(
                SimulationParameters(
                    drive_mlr = drives[i]
                    # amplitude_gradient=None,
                    # phase_lag_body=None
                )
            )
        network.step(i, time0, timestep)
        phases_log[i+1, :] = network.state.phases(iteration=i+1)
        amplitudes_log[i+1, :] = network.state.amplitudes(iteration=i+1)
        outputs_log[i+1, :] = network.get_motor_position_output(iteration=i+1)
        freqs_log[i+1, :] = network.robot_parameters.freqs
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
    generate_plots(times, phases_log, amplitudes_log, outputs_log, freqs_log, drives)


def generate_plots(times, phases_log, amplitudes_log, outputs_log, freqs_log, drives):
    fig, axs = plt.subplots(4, 1)
    
    plot_phase(phases_log)
    plot_amplitude(amplitudes_log)
    plot_output(outputs_log)
    plot_freq(times, freqs_log, axs)
    plot_drive(times, drives, axs)
    
    
    plt.show()

def plot_phase(phases_log):
    return
def plot_amplitude(amplitudes_log):
    return
def plot_output(outputs_log):
    return

def plot_freq(times, freqs_log, axs):
    print(freqs_log)
    #ax = fig.add_subplot(111)
    axs[2].plot(times, freqs_log, color='k')
    
    

def plot_drive(times, drives, axs):
    #ax = fig.add_subplot(111)
    supLimb= False
    supBody = False
    supAll = False
    for i,d in enumerate(drives):
        if d>1 and not supLimb:
            plt.vlines(x=times[i], ymin= 0, ymax=6, linestyle='--', color='grey')
            plt.text(0.2, 0.45,'Walking' ,horizontalalignment='center',
     verticalalignment='center', transform = axs[3].transAxes)
            supLimb = True
        if d>3 and not supBody:
            plt.vlines(x=times[i], ymin= 0, ymax=6, linestyle='--', color='grey')
            plt.text(0.6, 0.8,'Swimming' ,horizontalalignment='center',
     verticalalignment='center', transform = axs[3].transAxes)
            supBody = True
        elif d>5 and not supAll:
            plt.vlines(x=times[i], ymin= 0, ymax=6, linestyle='--', color='grey')
            supAll = True
    
    axs[3].plot(times,drives, 'k')
    axs[3].hlines([1,3,5],0,times[-1], color='orange')
    axs[3].set_xlabel('Time [s]')
    axs[3].set_ylabel('drive d')
    axs[3].set_xlim(0, times[-1])
    axs[3].set_ylim(0, 6)
    
    

def main(plot):
    """Main"""

    run_network(duration=5)

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main(plot=not save_plots())

