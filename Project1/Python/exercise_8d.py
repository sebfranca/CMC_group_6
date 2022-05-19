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

def exercise_8d1(timestep, duration=20):
    """Exercise 8d1"""
    drive_params = {
        "baseline": 3.5,
        "delta_turn": 1,
        "turn_start": 2,
        "turn_duration": 5,
        "direction": "right"}
    times = np.arange(0, duration, timestep)
    n_iterations = len(times)
    
    drive_mlr  = drive_params["baseline"]
    drive_offset_turn = drive_params["delta_turn"]
    
    start     = drive_params["turn_start"]
    duration  = drive_params["turn_duration"]
    
    turns = ["None" for i in range(n_iterations)]
    for i, t in enumerate(times):
        if t>=start and t<=start + duration:
            turns[i] = drive_params["direction"]
            
    #In order to turn, must give duration, timestep, drive_mlr, drive_offset_turn and turns.
            
    sim_parameters = SimulationParameters(
        duration=duration,  # Simulation duration in [s]
        timestep=timestep,  # Simulation timestep in [s]
        spawn_position=[0, 0, 0.1],  # Robot position in [m]
        spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
        drive_mlr=drive_mlr,  # An example of parameter part of the grid search
        drive_offset_turn = drive_offset_turn,
        turns = turns
    )
    
    filename = './logs/test8d'
    sim, data = simulation(
        sim_parameters=sim_parameters,  # Simulation parameters, see above
        arena='water',  # Can also be 'ground', give it a try!
                # fast=True,  # For fast mode (not real-time)
        #headless=True,  # For headless mode (No GUI, could be faster)
                # record=True,  # Record video
        )
    
    
   
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

if __name__ == '__main__':
    exercise_8d1(1e-2)