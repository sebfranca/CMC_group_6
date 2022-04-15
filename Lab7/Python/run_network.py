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
    
    phase_bias = np.zeros([20,20])
    for i in range(16):
        for j in range(16):
            if j == i+8 or j==i-8:
                phase_bias[i,j] = np.pi #i and j on same row
            
            elif i<8 and j<8: #i and j on the left
                if j==i+1: #j comes just after i
                    phase_bias[i,j] = -2*np.pi/16
                elif j==i-1: #j comes just before i
                    phase_bias[i,j] = 2*np.pi/16
                    
            elif (i>=8 and i<16) and (j>=8 and j<16): #i and j on the right
                if j==i-1:
                    phase_bias[i,j] = 2*np.pi/16
                elif j==i+1:
                    phase_bias[i,j] = -2*np.pi/16
    
    
    coupling = np.zeros([20,20])
    for i in range(16):
        for j in range(16):
            if i==j:
                coupling[i,j] = 1
            elif (j==i+8 or j==i-8) and j<16: #i and j left&right
                coupling[i,j] = 10
            elif (j==i+1 and j!=8) or (j==i-1 and j!=7) and j<16: #i and j are near each other
                coupling[i,j] = 10
            
    
    #Here, we must set: f_i, w_ij, phi_ij, a_i, R_i
    sim_parameters = SimulationParameters(
        drive=drive,
        amplitude_gradient=None,
        phase_lag=None,
        turn=None,
        freqs = np.ones(20),
        coupling_weights = coupling,
        phase_bias = phase_bias,
        rates = [0.25]*20,
        nominal_amplitudes = [0.75]*20,
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

    # Run network ODE and log data
    tic = time.time()
    for i, time0 in enumerate(times[1:]):
        if update:
            network.robot_parameters.update(
                SimulationParameters(
                    # amplitude_gradient=None,
                    # phase_lag=None
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
    plt.figure()
    plt.plot(times,np.cos(phases_log[:,0:3]) )
    plt.figure()
    plt.plot(times,np.cos(phases_log[:,0:2]) )
    plt.plot(times,np.cos(phases_log[:,8:10]))
    plt.figure()
    plt.plot(times,amplitudes_log)


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

