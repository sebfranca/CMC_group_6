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
from plot_results import main as makeplots

def exercise_8d1(timestep, duration=45):
    """Exercise 8d1"""
    turn_params = {
        "baseline": 3.5,
        "delta_turn": 0.5,
        "turn_starts": [10,30],
        "turn_durations": [10,5],
        "directions": ["right","left"]}
    times = np.arange(0, duration, timestep)
    n_iterations = len(times)
    
    drive_mlr  = turn_params["baseline"]
    drive_offset_turn = turn_params["delta_turn"]
    
    starts     = turn_params["turn_starts"]
    turn_durations  = turn_params["turn_durations"]
    
    turns = ["None" for i in range(n_iterations)]
    for turn_idx,start in enumerate(starts):
        turn_duration = turn_durations[turn_idx]
        for i, t in enumerate(times):
            if t>=start and t<=start + turn_duration:
                turns[i] = turn_params["directions"][turn_idx]
            
    #In order to turn, must give duration, timestep, drive_mlr, drive_offset_turn and turns.
            
    sim_parameters = SimulationParameters(
        duration=duration,  # Simulation duration in [s]
        timestep=timestep,  # Simulation timestep in [s]
        drive_mlr=drive_mlr,  # An example of parameter part of the grid search
        drive_offset_turn = drive_offset_turn,
        turns = turns
    )
    
    os.makedirs('./logs/ex_8d1/', exist_ok=True)
    filename = './logs/ex_8d1/simulation.{}'
    sim, data = simulation(
        sim_parameters=sim_parameters,  # Simulation parameters, see above
        arena='water',  # Can also be 'ground', give it a try!
        fast=True,  # For fast mode (not real-time)
        #headless=True,  # For headless mode (No GUI, could be faster)
                # record=True,  # Record video
        )
    data.to_file(filename.format('h5'), sim.iteration)
    with open(filename.format('pickle'), 'wb') as param_file:
        pickle.dump(sim_parameters, param_file)
    
    makeplots(plot=True, ex_id='8d1')

def exercise_8d2(timestep, duration=20):
    """Exercise 8d2"""
    # Use exercise_example.py for reference
    drive_mlr = 4
    
    
    sim_parameters = SimulationParameters(
        duration=duration,  # Simulation duration in [s]
        timestep=timestep,  # Simulation timestep in [s]
        
        drive_mlr=drive_mlr,  # An example of parameter part of the grid search
        backward = True
    )
    
    os.makedirs('./logs/ex_8d2/', exist_ok=True)
    filename = './logs/ex_8d2/simulation.{}'
    sim, data = simulation(
        sim_parameters=sim_parameters,  # Simulation parameters, see above
        arena='water',  # Can also be 'ground', give it a try!
        fast=True,  # For fast mode (not real-time)
        #headless=True,  # For headless mode (No GUI, could be faster)
                # record=True,  # Record video
        )
    data.to_file(filename.format('h5'), sim.iteration)
    with open(filename.format('pickle'), 'wb') as param_file:
        pickle.dump(sim_parameters, param_file)
    
    makeplots(plot=True, ex_id='8d2')

if __name__ == '__main__':
    exercise_8d1(1e-2)
    exercise_8d2(1e-2)