"""Exercise 9b"""

import numpy as np
from farms_core import pylog
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
import os
import pickle
from plot_results import main as makeplots


def exercise_9b(timestep, duration=20):
    """Exercise 9b"""
    #reach ground : head was located >1, becomes <1
    #reach water : head was located <1.5, becomes >1.5
    
    #Walk to swim
    sim_parameters = SimulationParameters(
        duration = duration,
        drive_mlr=4,
        amphibious = True,
        
        spawn_position=[0, 0, 0],
        spawn_orientation=[0, 0, 0],
    )
    sim, data = simulation(
        sim_parameters=sim_parameters,
        arena='amphibious',
        fast=True,
        record=True,
        record_path='walk2swim',  # or swim2walk
    )
    
    os.makedirs('./logs/ex_9b/', exist_ok=True)
    filename = './logs/ex_9b/walk2swim.{}'
    data.to_file(filename.format('h5'), sim.iteration)
    with open(filename.format('pickle'), 'wb') as param_file:
        pickle.dump(sim_parameters, param_file)
        
    makeplots(plot=True, ex_id='9b', grid_id="walk2swim")
    
    #Swim to walk
    sim_parameters = SimulationParameters(
        duration = duration,
        drive_mlr=4,
        amphibious = True,
        
        spawn_position=[2.5, 0, 0.0],
        spawn_orientation=[0, 0, np.pi],
    )
    _sim, _data = simulation(
        sim_parameters=sim_parameters,
        arena='amphibious',
        fast=True,
        record=True,
        record_path='swim2walk',
    )
    
    os.makedirs('./logs/ex_9b/', exist_ok=True)
    filename = './logs/ex_9b/swim2walk.{}'
    data.to_file(filename.format('h5'), sim.iteration)
    with open(filename.format('pickle'), 'wb') as param_file:
        pickle.dump(sim_parameters, param_file)
        
    makeplots(plot=True, ex_id='9b', grid_id="swim2walk")

if __name__ == '__main__':
    exercise_9b(timestep=1e-2)

