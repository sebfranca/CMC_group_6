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

def exercise_8d1(timestep, duration=20, update=True):
    """Exercise 8d1"""
    drive_params = {
        "baseline": 3.5,
        "delta_turn": 1,
        "turn_start": 2,
        "turn_duration": 5,
        "direction": "left"}
    times = np.arange(0, duration, timestep)
    n_iterations = len(times)
    
    drive_mlr  = drive_params["baseline"]*np.ones(len(times))
    drive_offset_turn = drive_params["delta_turn"]
    
    start     = drive_params["turn_start"]
    duration  = drive_params["turn_duration"]
    
    turns = None * np.ones(n_iterations)
    for i, t in enumerate(times):
        if t>=start and t<=start + duration:
            turns[i] = drive_params["direction"]
            
            
    sim_parameters = SimulationParameters(
        drive_mlr = drive_mlr,
        drive_offset_turn = drive_offset_turn,
    )        
    
    
    state = SalamandraState.salamandra_robotica_2(n_iterations)
    network = SalamandraNetwork(sim_parameters, n_iterations, state)
    osc_left = np.arange(8)
    osc_right = np.arange(8, 16)
    osc_legs = np.arange(16, 20)

    # Logs
    phases_log = np.zeros([
        n_iterations,
        len(network.state.phases(iteration=0))
    ])
    phases_log[0, :] = network.state.phases(iteration=0)
    amplitudes_log = np.zeros([
        n_iterations,
        len(network.state.amplitudes(iteration=0))
    ])
    amplitudes_log[0, :] = network.state.amplitudes(iteration=0)
    freqs_log = np.zeros([
        n_iterations,
        len(network.robot_parameters.freqs)
    ])
    freqs_log[0, :] = network.robot_parameters.freqs
    outputs_log = np.zeros([
        n_iterations,
        len(network.get_motor_position_output(iteration=0))
    ])
    outputs_log[0, :] = network.get_motor_position_output(iteration=0)
    
    tic = time.time()
    for i, time0 in enumerate(times[1:]):
        if update:
            network.robot_parameters.update(
                SimulationParameters(
                    drive_mlr = drive_mlr,
                    turn = turns[i]
                )
            )
        network.step(i, time0, timestep)
        phases_log[i+1, :] = network.state.phases(iteration=i+1)
        amplitudes_log[i+1, :] = network.state.amplitudes(iteration=i+1)
        outputs_log[i+1, :] = network.get_motor_position_output(iteration=i+1)
        freqs_log[i+1, :] = network.robot_parameters.freqs
    
    


def exercise_8d2(timestep):
    """Exercise 8d2"""
    # Use exercise_example.py for reference
    pass


if __name__ == '__main__':
    exercise_8d1(timestep=1e-2)
    exercise_8d2(timestep=1e-2)

