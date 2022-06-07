"""Simulation parameters"""

import numpy as np


class SimulationParameters:
    """Simulation parameters"""

    def __init__(self, **kwargs):
        super(SimulationParameters, self).__init__()
        # Default parameters
        self.n_body_joints = 8
        self.n_legs_joints = 4
        self.duration = kwargs.get("duration",40)
        self.timestep = kwargs.get("timestep",1e-2)
        self.times = np.arange(0,self.duration,self.timestep)
        self.n_iterations = len(self.times)
        self.initial_phases = None
        self.phase_lag_body = None
        self.amplitude_gradient = None
        self.spawn_position=[0, 0, 0.1]  # Robot position in [m]
        self.spawn_orientation=[0, 0, 0]  # Orientation in Euler angles [rad]
        self.drive_mlr = 0#2
        
        #Parameters used to force the definition of matrices, 
        #in robot_parameters.py
        self.exercise_8b = False
        self.exercise_8c = False
        self.decoupled = False
        self.exercise_8f = False
        self.exercise_9a_phase = False
        self.exercise_9a_amplitude = False
        self.nominal_amplitude_parameters = np.zeros(2)
        self.nominal_amplitudes = np.zeros(20)
        self.phase_bias = np.zeros([20,20])
        self.l2b_phase = 0
        self.body_amplitude = 0
        
        #Parameters used for turning and going backward
        self.turns = ["None" for i in range(self.n_iterations)] #all turn instructions
        self.turn_instruction = self.turns[0] #turn instruction of current iteration
        self.drive_offset_turn = 0.5
        self.backward = False
        
        #Parameters used to adjust CPG vs sensory feedback
        self.fb_gain = 2
        self.b2b_same_coupling = 10
        self.cpg_active = True
        self.fb_active = False

        # Disruptions
        self.set_seed = False
        self.randseed = 0
        self.n_disruption_couplings = 0
        self.n_disruption_oscillators = 0
        self.n_disruption_sensors = 0
        
        #Amphibious arena
        self.amphibious = False

        # Update object with provided keyword arguments
        # NOTE: This overrides the previous declarations
        self.__dict__.update(kwargs)

