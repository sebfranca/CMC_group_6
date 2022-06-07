"""Exercise 9a"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
from plot_results import main as makeplots

def exercise_9a(timestep, duration = 20):
    """Exercise 9a"""
# =============================================================================
#     for grid_id in ["walking","swimming"]:
#         if grid_id=="walking":
#             drive_mlr = 2.5
#             arena = 'ground'
#         else:
#             drive_mlr = 4
#             arena = 'water'
#         
#         sim_params = SimulationParameters(
#             duration = duration,
#             timestep = timestep,
#             
#             drive_mlr = drive_mlr
#             )
#     
#         # Use exercise_example.py for reference
#         sim, data = simulation(
#             sim_parameters = sim_params,
#             fast = True,
#             arena = arena)
#         
#         os.makedirs('./logs/ex_9a_simple/', exist_ok=True)
#         filename = './logs/ex_9a_simple/{}.{}'
#         data.to_file(filename.format(grid_id,'h5'), sim.iteration)
#         with open(filename.format(grid_id,'pickle'), 'wb') as param_file:
#             pickle.dump(sim_params, param_file)
#             
#         makeplots(plot=True, ex_id='9a_simple', grid_id=grid_id)
# =============================================================================
        
        
    #Run grid search number 1
    grid_id = "phase"
    
    sim_params = [SimulationParameters(
        duration = duration,
        timestep = timestep,
        
        exercise_9a_phase = True,
        l2b_phase = phase,
        
        drive_mlr = 2.5
        )
        for phase in np.linspace(0,2*np.pi,10)
        ]
    
    for sim_i, sim_p in enumerate(sim_params):
        sim, data = simulation(
            sim_parameters = sim_p,
            fast = True,
            headless = True,
            arena = 'ground')
        
        os.makedirs('./logs/ex_9a_grid/grid{}/'.format(grid_id), exist_ok=True)
        filename = './logs/ex_9a_grid/grid{}/simulation_{}.{}'
        data.to_file(filename.format(grid_id,sim_i,'h5'), sim.iteration)
        with open(filename.format(grid_id,sim_i,'pickle'), 'wb') as param_file:
            pickle.dump(sim_p, param_file)
            
    makeplots(plot=True, ex_id='9a_grid', grid_id=grid_id)
    
    #Run grid search number 2
    grid_id = "amplitude"
    
    sim_params = [SimulationParameters(
        duration = duration,
        timestep = timestep,
        
        exercise_9a_amplitude = True,
        body_amplitude = amp,
        
        drive_mlr = 2.5
        )
        for amp in np.linspace(0,0.75,10)
        ]
    
    for sim_i, sim_p in enumerate(sim_params):
        sim, data = simulation(
            sim_parameters = sim_p,
            fast = True,
            headless = True,
            arena = 'ground')
        
        print("Amplitude : " + str(sim_p.body_amplitude))
        
        os.makedirs('./logs/ex_9a_grid/grid{}/'.format(grid_id), exist_ok=True)
        filename = './logs/ex_9a_grid/grid{}/simulation_{}.{}'
        data.to_file(filename.format(grid_id,sim_i,'h5'), sim.iteration)
        with open(filename.format(grid_id,sim_i,'pickle'), 'wb') as param_file:
            pickle.dump(sim_p, param_file)
            
    makeplots(plot=True, ex_id='9a_grid', grid_id=grid_id)
    
    
if __name__ == '__main__':
    exercise_9a(timestep=1e-2)

