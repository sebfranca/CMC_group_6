"""Exercise 8b"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters


def exercise_8b(timestep):
    """Exercise 8b"""
# =============================================================================
#     Now that you have implemented the controller, it is time to run experiments to study its behaviour.
# How does phase lag and oscillation amplitude influence the speed and energy? Use the provided exercise_8b.py 
#to run a grid search to explore the robot behavior for different combinations of amplitudes
# and phase lags. Use plot_results.py to load and plot the logged data from the simulation. Include
# 2D/3D plots showing your grid search results and discuss them. How do your findings compare to the
# wavelengths observed in the salamander?
# Hint 1: To use the grid search, check out the example provided in exercise_example.py. This
# function takes the desired parameters as a list of SimulationParameters objects (found in sim-
# ulation_parameters.py) and runs the simulation. Note that the results are logged as simulation_#.h5 in a specified log folder. After the grid search finishes, the simulation will close.
# • Hint 2: An example of how to load and visualise grid search results is already implemented
# in plot_results.py::main(). Pay attention to the name of the folder and the log files you are
# loading. Before starting a new grid search, change the name of the logs destination folder where
# the results will be stored. In case a grid search failed, it may be safer to delete the previous logs
# to avoid influencing new results by mistake.
# • Hint 3: Estimate how long it will take to finish the grid search. Our suggestion is to choose
# wisely lower and upper limits of parameter vectors and choose a reasonable number of samples.
# To speed-up a simulation, make sure to run the simulation in fast mode and without GUI as
# shown in exercise_example.py or using –fast and –headless in the Python command line (Use
# –help for more information).
# • Hint 4: Energy can be estimated by integrating the product of instantaneous joint velocities and
# torques. Feel free to propose your own energy metrics, just make sure to include a justification
# for the one chosen.
# =============================================================================
    
    phase_lag_params = {
        'b2b_same' : [10],
        'b2b_opp' : [10],
        'l2l_same' : [10],
        'l2l_opp' : [10],
        'l2b' : [30]
        }
    amplitude_params = {
        'amplitude_limbs' : [1],
        'amplitude_body' : [1]
        }
    
    phase_lag = []
    amplitudes = []
    
    for i in range(len(phase_lag_params['l2b'])):
        phase_lag.append(make_matrix(phase_lag_params,i))
        amplitudes.append(make_amplitudes(amplitude_params,i))
    
    #make a 1D grid that covers all combinations
    grid = {'amplitudes':[],'phase_lags':[]}
    for i,amp in enumerate(amplitudes):
        for j,lag in enumerate(phase_lag):
            grid['amplitudes'].append(amp)
            grid['phase_lags'].append(lag)
    
    # parameter_set = [SimulationParameters(
    #     duration = 10,
    #     timestep = timestep,
    #     spawn_position=[0,0,0.1],
    #     spawn_orientation=[0,0,0],
    #     drive = 0.5,
    #     amplitudes = grid['amplitudes'][i],
    #     phase_lag = grid['phase_lags'][i]
    #     )
    #     for i in range(len(grid['amplitudes']))
    #     ]
    
    
    
    # os.makedirs('./logs/example/', exist_ok=True)
    # for simulation_i, sim_parameters in enumerate(parameter_set):
    #     filename = './logs/example/simulation_{}.{}'
    #     sim, data = simulation(
    #         sim_parameters=sim_parameters,  # Simulation parameters, see above
    #         arena='water',  # Can also be 'ground', give it a try!
    #         # fast=True,  # For fast mode (not real-time)
    #         # headless=True,  # For headless mode (No GUI, could be faster)
    #         # record=True,  # Record video
    #     )
    #     # Log robot data
    #     data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
    #     # Log simulation parameters
    #     with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
    #         pickle.dump(sim_parameters, param_file)


    
def make_matrix(params,i):
    """
    Creates the coupling matrix.
    
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
    
    
    body_segments = 8
    limbs = 4
    
    coupling = np.zeros([20,20])
    
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
        for j in range(20):
            if i==j: pass
            
            elif isBody(i) and isBody(j):
                if bodyOnSameSide(i, j) and abs(i-j)==1:
                    coupling[i,j] = b2b_same
                elif bodyOnOppSide(i, j) and abs(i-j)==8:
                    coupling[i,j] = b2b_opp
                    
            elif isLimb(i) and isLimb(j):
                if limbOnSameSide(i, j):
                    coupling[i,j] = l2l_same
                elif limbOnOppSide(i, j) and abs(i-j)==2:
                    coupling[i,j] = l2l_opp
            
            elif isBody(i) and isLimb(j):
                if bodyLimbOnSameSide(i, j):
                    if (j in frontLimbs and i in frontBodies) or (j in backLimbs and i in backBodies):
                        coupling[j,i] = l2b

    return coupling


if __name__ == '__main__':
    exercise_8b(timestep=1e-2)

