"""Exercise 9a"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters


def exercise_9a(timestep, duration = 50):
    """Exercise 9a"""
    drive_mlr = 2.5
    
    sim_params = SimulationParameters(
        duration = duration,
        timestep = timestep,
        
        drive_mlr = drive_mlr
        )

    # Use exercise_example.py for reference
    sim, data = simulation(
        sim_parameters = sim_params,
        arena = 'ground')


if __name__ == '__main__':
    exercise_9a(timestep=1e-2)

