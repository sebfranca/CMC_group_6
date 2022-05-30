"""Exercise 8e"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
from plot_results import main as makeplots

def exercise_8e1(timestep, duration = 25):
    """Exercise 8e1"""
    drive_mlr = 4
   
    sim_params = SimulationParameters(
       drive_mlr = drive_mlr,
       decoupled = True,
       duration = duration,
       timestep = timestep
       )
   
    sim, data = simulation(sim_params, arena='water',)
    
    
    os.makedirs('./logs/ex_8e1/', exist_ok=True)
    filename = './logs/ex_8e1/simulation.{}'
    data.to_file(filename.format('h5'), sim.iteration)
    with open(filename.format('pickle'), 'wb') as param_file:
        pickle.dump(sim_params, param_file)
    
    #makeplots(plot=True, ex_id='8e1')
    

def exercise_8e2(timestep, duration = 25):
    """Exercise 8e2"""

    # Use exercise_example.py for reference
    drive_mlr = 4
   
    sim_params = SimulationParameters(
       drive_mlr = drive_mlr,
       decoupled = True,
       fb_active = True,
       fb_gain = 10,
       duration = duration,
       timestep = timestep,
       
       set_seed = True,
       randseed = 0
       )
   
    sim, data = simulation(sim_params, arena='water')

    os.makedirs('./logs/ex_8e2/', exist_ok=True)
    filename = './logs/ex_8e2/simulation.{}'
    data.to_file(filename.format('h5'), sim.iteration)
    with open(filename.format('pickle'), 'wb') as param_file:
        pickle.dump(sim_params, param_file)
    
    #makeplots(plot=True, ex_id='8e2')

if __name__ == '__main__':
    #exercise_8e1(timestep=1e-2)
    exercise_8e2(timestep=1e-2)

