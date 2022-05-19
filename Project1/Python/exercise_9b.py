"""Exercise 9b"""

import numpy as np
from farms_core import pylog
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters


def exercise_9b(timestep):
    """Exercise 9b"""

    # Use exercise_example.py for reference
    # Additional hints:
    # sim_parameters = SimulationParameters(
    #     ...,
    #     spawn_position=[0, 0, 0.1],
    #     spawn_orientation=[0, 0, 0],
    #     # Or
    #     spawn_position=[4, 0, 0.0],
    #     spawn_orientation=[0, 0, np.pi],
    # )
    # _sim, _data = simulation(
    #     sim_parameters=sim_parameters,
    #     arena='amphibious',
    #     fast=True,
    #     record=True,
    #     record_path='walk2swim',  # or swim2walk
    # )
    pass


if __name__ == '__main__':
    exercise_9b(timestep=1e-2)

