"""Exercise example"""

import os
import pickle
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
import numpy as np

def exercise_example(timestep):
    """Exercise example"""
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
       


    # phase_bias = np.zeros([20,20])
    # for i in range(16):
    #     for j in range(16):
    #         if j == i+8 or j==i-8:
    #             phase_bias[i,j] = np.pi #i and j on same row
            
    #         elif i<8 and j<8: #i and j on the left
    #             if j==i+1: #j comes just after i
    #                 phase_bias[i,j] = -2*np.pi/body_segments
    #             elif j==i-1: #j comes just before i
    #                 phase_bias[i,j] = 2*np.pi/body_segments
                    
    #         elif (i>=8 and i<16) and (j>=8 and j<16): #i and j on the right
    #             if j==i-1:
    #                 phase_bias[i,j] = 2*np.pi/body_segments
    #             elif j==i+1:
    #                 phase_bias[i,j] = -2*np.pi/body_segments
    
    
    # coupling = np.zeros([20,20])
    # for i in range(16):
    #     for j in range(16):
    #         if i==j:
    #             coupling[i,j] = 1
    #         elif (j==i+8 or j==i-8) and j<16: #i and j left&right
    #             coupling[i,j] = 10
    #         elif (j==i+1 and j!=8) or (j==i-1 and j!=7) and j<16: #i and j are near each other
    #             coupling[i,j] = 10


                
    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=100,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=4*np.ones_like(np.arange(0, 100, timestep)),  # An example of parameter part of the grid search
            amplitudes=[1, 2, 3],  # Just an example
            phase_lag=0,  # or np.zeros(n_joints) for example
            turn=0,  # Another example
            coupling_weights = coupling,
            rates = [20]*20,
            phase_bias = phase_bias,
            # ...
        )
    ]

    # Grid search
    os.makedirs('./logs/example/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/example/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'ground', give it a try!
            # fast=True,  # For fast mode (not real-time)
            # headless=True,  # For headless mode (No GUI, could be faster)
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)


if __name__ == '__main__':
    exercise_example(timestep=1e-2)

