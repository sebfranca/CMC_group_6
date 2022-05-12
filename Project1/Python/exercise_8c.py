"""Exercise 8c"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters


def exercise_8c(timestep=1e-2, duration=20):
    """Exercise 8c"""
    nominal_amplitude_parameters = []
    grid_id = 100
    
    param_range = np.linspace(0, 1, 6) #change here
    for i in param_range:
        for j in param_range:
            nominal_amplitude_parameters.append([i,j])
    
    #make a 1D grid that covers all combinations
    grid = {'nominal_amplitude_parameters':[]}
    for idx, param in enumerate(nominal_amplitude_parameters):
            grid['nominal_amplitude_parameters'].append(param)
    for i in range(len(grid['nominal_amplitude_parameters'])):
            print(grid['nominal_amplitude_parameters'][i])
    
    parameter_set = [SimulationParameters(
        duration=duration,  # Simulation duration in [s]
        timestep=timestep,  # Simulation timestep in [s]
        spawn_position=[0, 0, 0.1],  # Robot position in [m]
        spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
        
        
        drive_mlr = 3.5,
        nominal_amplitude_parameters = grid['nominal_amplitude_parameters'][i], 
        exercise_8c = True

        )
        for i in range(len(grid['nominal_amplitude_parameters']))
        ]
    

    
    
    
    os.makedirs('./logs/grid{}/'.format(grid_id), exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/grid{}/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'ground', give it a try!
            fast=True,  # For fast mode (not real-time)
            headless=True,  # For headless mode (No GUI, could be faster)
            # record=True,  # Record video
        )
        # Log robot data
        data.to_file(filename.format(grid_id, simulation_i, 'h5'), sim.iteration)

        # Log simulation parameters
        with open(filename.format(grid_id,simulation_i, 'pickle'), 'wb') as param_file:

            pickle.dump(sim_parameters, param_file)




if __name__ == '__main__':
    exercise_8c(timestep=1e-2)

