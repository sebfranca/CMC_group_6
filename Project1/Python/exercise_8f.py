"""Exercise 8f"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
from plot_results import main as makeplots

def exercise_8f(timestep, duration=20):
    """Exercise 8f"""
    grid_id = '_test'
    
    min_c = 0
    max_c = 18
    min_gain = 0
    max_gain = 4
    
    num_c = 5
    num_gain = 5
    
    b2b_same_couplings = np.linspace(min_c,max_c,num_c)
    fb_gains = np.linspace(min_gain,max_gain,num_gain)
   
    parameter_set = [SimulationParameters(
        duration=duration,  # Simulation duration in [s]
        timestep=timestep,  # Simulation timestep in [s]
        set_seed = True,
        
        drive_mlr = 4,
        fb_gain = gain,
        b2b_same_coupling = coupling,
        
        fb_active = True,
        exercise_8f = True,
        )
        for gain in fb_gains
        for coupling in b2b_same_couplings
        ]
    
    #Save results
    os.makedirs('./logs/ex_8f/grid{}/'.format(grid_id), exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/ex_8f/grid{}/simulation_{}.{}'
        print("Feedback gain :" + str(sim_parameters.fb_gain))
        print("Coupling: "  + str(sim_parameters.b2b_same_coupling))
        
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
    
    
    makeplots(plot=True, ex_id = '8f', grid_id=grid_id)

if __name__ == '__main__':
    exercise_8f(timestep=1e-2)

