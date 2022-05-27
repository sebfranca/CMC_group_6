"""Exercise 8g"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
from plot_results import main as makeplots

def init_disruptions(seed=0):
    np.random.seed(seed)
    
    disruptions = [
        {'n_disruption_couplings':0, 'n_disruption_oscillators':0, 'n_disruption_sensors':0}
        ]
    
    total_disruptions = [3,6,9]
    
    #muted coupling
    for n_disruption_couplings in total_disruptions:
        disruptions.append(
            {'n_disruption_couplings':n_disruption_couplings,
             'n_disruption_oscillators':0,
             'n_disruption_sensors':0}
            )
    #disfunctional oscillators
    for n_disruption_oscillators in total_disruptions:
        disruptions.append(
            {'n_disruption_couplings':0,
             'n_disruption_oscillators':n_disruption_oscillators,
             'n_disruption_sensors':0}
            )
    #disrupted sensors
    for n_disruption_sensors in total_disruptions:
        disruptions.append(
            {'n_disruption_couplings':0,
             'n_disruption_oscillators':0,
             'n_disruption_sensors':n_disruption_sensors}
            )
        
    # #mixed (1 of each)
    # for k in [1,2,3]:
    #     disruptions.append(
    #         {'n_disruption_couplings':k,
    #          'n_disruption_oscillators':k,
    #          'n_disruption_sensors':k}
    #         )
        
    #mixed (random)
    for k in total_disruptions:
        remaining_disruptions = k
        dic = {'n_disruption_couplings':0,
         'n_disruption_oscillators':0,
         'n_disruption_sensors':0}
        
        while remaining_disruptions>0:
            d = np.random.choice(['c','o','s'])
            if d=='c':
                dic['n_disruption_couplings'] += 1
            elif d=='o':
                dic['n_disruption_oscillators'] += 1
            elif d=='s':
                dic['n_disruption_sensors'] += 1
            remaining_disruptions -=1
        
        disruptions.append(dic)
        
    return disruptions

def exercise_8g1(timestep,  duration=20, seed=0, drive_mlr=4, disruptions = init_disruptions()):
    """Exercise 8g1: CPG only"""
    print('Running 8g1: CPG only')
    
    
    np.random.seed(seed) #for repeatability: necessary or not?
    disruptions = init_disruptions()
    
    parameter_set = [SimulationParameters(
       duration = duration,
       timestep = timestep,
       set_seed = True,
       randseed = seed,
       
       n_disruption_couplings = disruption['n_disruption_couplings'],
       n_disruption_oscillators = disruption['n_disruption_oscillators'],
       n_disruption_sensors = disruption['n_disruption_sensors'],
       
       drive_mlr = drive_mlr,
       fb_gain = 2,
       fb_active = False,
       decoupled = False,
       )
        for disruption in disruptions
        ]
   
    
    #Save results
    os.makedirs('./logs/ex_8g1/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        c = sim_parameters.n_disruption_couplings
        o = sim_parameters.n_disruption_oscillators
        s = sim_parameters.n_disruption_sensors
        
        print("Couplings: {} disruptions".format(c))
        print("Oscillators: {} disruptions".format(o))
        print("Sensors: {} disruptions".format(s))
        
        filename = './logs/ex_8g1/simulation_{}.{}'
        
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'ground', give it a try!
            #fast=True,  # For fast mode (not real-time)
            #headless=True,  # For headless mode (No GUI, could be faster)
            # record=True,  # Record video
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)
    
    makeplots(plot=True, ex_id='8g1')


def exercise_8g2(timestep, duration=20, seed=0, drive_mlr=4, disruptions = init_disruptions()):
    """Exercise 8g2: decoupled"""
    print('Running 8g2: decoupled')
    drive_mlr = 4
    
    disruptions = init_disruptions()
    
    parameter_set = [SimulationParameters(
       duration = duration,
       timestep = timestep,
       set_seed = True,
       randseed = seed,
       
       n_disruption_couplings = disruption['n_disruption_couplings'],
       n_disruption_oscillators = disruption['n_disruption_oscillators'],
       n_disruption_sensors = disruption['n_disruption_sensors'],
       
       drive_mlr = drive_mlr,
       fb_gain = 2,
       fb_active = True,
       decoupled = True,
       )
        for disruption in disruptions
        ]
   
    
    #Save results
    os.makedirs('./logs/ex_8g2/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        c = sim_parameters.n_disruption_couplings
        o = sim_parameters.n_disruption_oscillators
        s = sim_parameters.n_disruption_sensors
        
        print("Couplings: {} disruptions".format(c))
        print("Oscillators: {} disruptions".format(o))
        print("Sensors: {} disruptions".format(s))
        
        filename = './logs/ex_8g2/simulation_{}.{}'
        
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'ground', give it a try!
            #fast=True,  # For fast mode (not real-time)
            #headless=True,  # For headless mode (No GUI, could be faster)
            # record=True,  # Record video
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)
    
    makeplots(plot=True, ex_id='8g2')

def exercise_8g3(timestep, duration=20, seed=0, drive_mlr=4, disruptions = init_disruptions()):
    """Exercise 8g3: combined"""
    print('Running 8g3: combined')
    
    disruptions = init_disruptions()
    
    parameter_set = [SimulationParameters(
       duration = duration,
       timestep = timestep,
       set_seed = True,
       randseed = seed,
       
       n_disruption_couplings = disruption['n_disruption_couplings'],
       n_disruption_oscillators = disruption['n_disruption_oscillators'],
       n_disruption_sensors = disruption['n_disruption_sensors'],
       
       drive_mlr = drive_mlr,
       fb_gain = 2,
       fb_active = True,
       decoupled = False,
       )
        for disruption in disruptions
        ]
   
    
    #Save results
    os.makedirs('./logs/ex_8g3/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        c = sim_parameters.n_disruption_couplings
        o = sim_parameters.n_disruption_oscillators
        s = sim_parameters.n_disruption_sensors
        
        print("Couplings: {} disruptions".format(c))
        print("Oscillators: {} disruptions".format(o))
        print("Sensors: {} disruptions".format(s))
        
        filename = './logs/ex_8g3/simulation_{}.{}'
        
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'ground', give it a try!
            #fast=True,  # For fast mode (not real-time)
            #headless=True,  # For headless mode (No GUI, could be faster)
            # record=True,  # Record video
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)
    
    makeplots(plot=True, ex_id='8g3')


if __name__ == '__main__':
    seed = 0
    drive_mlr = 4
    disruptions = init_disruptions(seed)
    
    exercise_8g1(timestep=1e-2, drive_mlr = drive_mlr, seed=seed, disruptions=disruptions)
    exercise_8g2(timestep=1e-2, drive_mlr = drive_mlr, seed=seed,disruptions=disruptions)
    exercise_8g3(timestep=1e-2, drive_mlr = drive_mlr, seed=seed,disruptions=disruptions)

