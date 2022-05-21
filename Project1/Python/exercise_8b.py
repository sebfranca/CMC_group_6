"""Exercise 8b"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
from plot_results import main as makeplots

def exercise_8b(timestep=1e-2, duration=20):
    """Exercise 8b"""
    grid_id = 13 #to avoid overwriting, identify each grid w/ a number
      
    gridsize_phase = 10
    gridsize_amp = 10
    
    phase_lag_params = {
        'b2b_same' : np.linspace(0,3*np.pi/8,gridsize_phase),
        'b2b_opp' : np.linspace(0,0,gridsize_phase),
        'l2l_same' : np.linspace(16/3*0,48/3*np.pi/8,gridsize_phase),
        'l2l_opp' : np.linspace(16/3*0,48/3*np.pi/8,gridsize_phase),
        'l2b' : np.linspace(0,0,gridsize_phase)
        }
    amplitude_params = {
        'amplitude_limbs' : np.linspace(0.2,0.9,gridsize_amp),
        'amplitude_body' : np.linspace(0.2,0.9,gridsize_amp)
        }
    
    
    phase_lag = []
    amplitudes = []
    
    for i in range(len(phase_lag_params['l2b'])):
        phase_lag.append(make_matrix(phase_lag_params,i, couplingM=False))
        amplitudes.append(make_amplitudes(amplitude_params,i))
    
    parameter_set = [SimulationParameters(
        duration=duration,  # Simulation duration in [s]
        timestep=timestep,  # Simulation timestep in [s]
        spawn_position=[0, 0, 0.1],  # Robot position in [m]
        spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
        drive_mlr = 4,
        nominal_amplitudes = nom_amp[0],
        phase_bias = lag,
        exercise_8b = True
        )
        for nom_amp in amplitudes
        for lag in phase_lag
        ]
    
    
    os.makedirs('./logs/ex_8b/grid{}/'.format(grid_id), exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/ex_8b/grid{}/simulation_{}.{}'
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
    
    
    makeplots(plot=True, ex_id = '8b', grid_id=grid_id)
    

    
def make_matrix(params,i, couplingM=False):
    """
    Creates the phase bias matrix.
    
    Parameters
    ----------------
    *coupling or phase lag values:
        b2b_ : body to body
        l2l_ : limb to limb
        l2b : limb to body
        
        _same : on the same side (left or right)
        _opp : opposite sides (left or right)
    """
    b2b_same = params['b2b_same'][i]
    b2b_opp = params['b2b_opp'][i]
    l2l_same = params['l2l_same'][i]
    l2l_opp = params['l2l_opp'][i]
    l2b = params['l2b'][i]
    
    
    matrix = np.zeros([20,20])
    
    isLimb = lambda j: j in [16,17,18,19]
    isBody = lambda i: i in [i for i in range(16)]
        
    limbOnLeft = lambda i: i%2 ==0
    limbOnRight = lambda i: i%2==1
    bodyOnLeft = lambda j: j in [i for i in range(8)]
    bodyOnRight = lambda j: j in [i for i in range(8,16)]
    
    limbOnSameSide = lambda i,j : (limbOnLeft(i) and limbOnLeft(j)) or (limbOnRight(i) and limbOnRight(j))
    limbOnOppSide = lambda i,j : (limbOnRight(i) and limbOnLeft(j)) or (limbOnLeft(i) and limbOnRight(j))
    bodyOnSameSide = lambda i,j : (bodyOnLeft(i) and bodyOnLeft(j)) or (bodyOnRight(i) and bodyOnRight(j))
    bodyOnOppSide = lambda i,j : (bodyOnRight(i) and bodyOnLeft(j)) or (bodyOnLeft(i) and bodyOnRight(j))
    bodyLimbOnSameSide = lambda i,j : (bodyOnLeft(i) and limbOnLeft(j)) or (bodyOnRight(i) and limbOnRight(j))
    bodyLimbOnOppSide = lambda i,j : (bodyOnRight(i) and limbOnLeft(j)) or (bodyOnLeft(i) and limbOnRight(j))
    
    frontLimbs = [16,17]
    backLimbs = [18,19]
    frontBodies = [0,1,2,3, 8,9,10,11]
    backBodies = [4,5,6,7, 12,13,14,15]
    
    for i in range(20):
        if isBody(i):
            j = i+1
            if bodyOnSameSide(i, j):
                matrix[i,j] = b2b_same
                
                if couplingM: matrix[j,i] = b2b_same
                else: matrix[j,i] = -b2b_same
            
            j = i+8
            if bodyOnOppSide(i, j):
                matrix[i,j] = b2b_opp
                matrix[j,i] = b2b_opp
            
        if isLimb(i):
            for j in range(i+1,i+4):
                if j<20 and limbOnSameSide(i, j) and i in frontLimbs and j in backLimbs:
                    matrix[i,j] = l2l_same
                    matrix[j,i] = l2l_same
                elif j<20 and limbOnOppSide(i, j) and ((i in frontLimbs and j in frontLimbs) or (i in backLimbs and j in backLimbs)):
                    matrix[i,j] = l2l_opp
                    matrix[j,i] = l2l_opp
            for j in range(16):
                if (i in frontLimbs and j in frontBodies) or (i in backLimbs and j in backBodies):
                    if bodyLimbOnSameSide(j,i):
                        matrix[i,j] = l2b
    
    return matrix

def make_amplitudes(params,i):
    amp_limb = params['amplitude_limbs'][i]
    amp_body = params['amplitude_body'][i]
    
    amplitudes = np.hstack((amp_body*np.ones([1,16]),amp_limb*np.ones([1,4])))
    return amplitudes


if __name__ == '__main__':
    exercise_8b(timestep=1e-2)

