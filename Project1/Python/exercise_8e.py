"""Exercise 8e"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters


def exercise_8e1(timestep, duration = 10):
    """Exercise 8e1"""
    drive_mlr = 4
    
    sim_params = SimulationParameters(
        drive_mlr = drive_mlr,
        cpg_active = False,
        fb_active = True,
        duration = duration,
        timestep = timestep
        )
    
    sim, data = simulation(sim_params, arena='water')


def exercise_8e2(timestep):
    """Exercise 8e2"""

    # Use exercise_example.py for reference
    pass


if __name__ == '__main__':
    exercise_8e1(timestep=1e-2)
    exercise_8e2(timestep=1e-2)

