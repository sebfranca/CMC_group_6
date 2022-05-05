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


def run_network(duration, update=False, drive=1, timestep=1e-2):
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
    times = np.arange(0, duration, timestep)
    n_iterations = len(times)
    
    if type(drive)==int or type(drive)==float:
        drive = drive * np.ones_like(times)
    elif len(drive)!=len(times):
        print("Wrong length of drive array")
    
    
    
    body_segments = 8
    limbs = 2
    coupling = np.zeros([20,20])
    front_limbs_idxs = 16+np.linspace(0,1,2)
    
    ## Coupling
    # Coupling weights of body oscillators 
    for i in range(2*body_segments):
        for j in range(2*body_segments):
            if i!=j:
                #No intraoscillator couplings
                if ((j==i+1 and j!=body_segments) or (j==i-1 and j!=body_segments-1)):
                    coupling[i,j] = 10 
                elif j-body_segments==i or j+body_segments==i:
                    coupling[i,j] = 10 
                    
    # Coupling weights of limb oscillators
    for i in range(2*limbs):
        for j in range(2*limbs):
            if i!=j:
                if (j==i-1 or j==i+1) and not(i==1 and j==2) and not(i==2 and j==1) or i%limbs==j%limbs:
                    coupling[i+2*body_segments, j+2*body_segments] = 10 
                    
    # Coupling between body and limbs
    for i in front_limbs_idxs:
        for j in range(round(body_segments/2)):
            if i%2 == 0:
                if j<body_segments/2:
                    coupling[int(i), j] = 30
                    coupling[int(i)+2, j+round(body_segments/2)] = 30
            else:
                coupling[int(i), j+body_segments] = 30
                coupling[int(i)+2, j+round(1.5*body_segments)] = 30
    
    ## Phase bias
    #Phase biases of body
    phase_bias = np.zeros([20,20])
    for i in range(2*body_segments):
        for j in range(2*body_segments):
            if i!=j:
               if ((j==i+1 and j!=body_segments) or (j==i-1 and j!=body_segments-1)):
                    if j==i+1:
                        phase_bias[i,j] = -2*np.pi/body_segments 
                    elif j==i-1: 
                        phase_bias[i,j] = 2*np.pi/body_segments 
                        
               elif j-body_segments==i or j+body_segments==i:
                    phase_bias[i,j] = np.pi #i and j on same row
    
    # Phase biases of limbs
    for i in range(2*limbs):
        for j in range(2*limbs):
            if i!=j:
                if (j==i-1 or j==i+1) and not(i==1 and j==2) and not(i==2 and j==1) or i%limbs==j%limbs:
                    phase_bias[i+2*body_segments, j+2*body_segments] = np.pi
                    
    
    
    #Here, we must set: f_i, w_ij, phi_ij, a_i, R_i
    sim_parameters = SimulationParameters(
        drive=drive,
        amplitude_gradient=None,
        phase_lag=None,
        turn=None,
        freqs = 0.4*np.ones(20),
        coupling_weights = coupling,
        phase_bias = phase_bias,
        rates = [20]*20,
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

