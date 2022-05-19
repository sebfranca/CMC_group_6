"""Simulation parameters"""

import numpy as np


class SimulationParameters:
    """Simulation parameters"""

    def __init__(self, **kwargs):
        super(SimulationParameters, self).__init__()
        # Default parameters
        self.n_body_joints = 8
        self.n_legs_joints = 4
        self.duration = kwargs.get("duration",30)
        self.timestep = kwargs.get("timestep",1e-2)
        self.times = np.arange(0,self.duration,self.timestep)
        self.n_iterations = len(self.times)
        self.initial_phases = None
        self.phase_lag_body = None
        self.amplitude_gradient = None
        self.spawn_position=[0, 0, 0.1]  # Robot position in [m]
        self.spawn_orientation=[0, 0, 0]  # Orientation in Euler angles [rad]
        
        
        self.drive_mlr = 2
        self.exercise_8b = False
        self.exercise_8c = False
        self.nominal_amplitude_parameters = np.zeros(2)
        
        
        
        self.turns = ["None" for i in range(self.n_iterations)]
        self.turn_instruction = self.turns[0]
        self.drive_offset_turn = 0.5
        self.backward = False
        
        
        self.nominal_amplitudes = np.zeros(20)
        self.phase_bias = np.zeros([20,20])
        
        # Feel free to add more parameters (ex: MLR drive)
        # self.drive_mlr = ...
        # ...

        # Disruptions
        self.set_seed = False
        self.randseed = 0
        self.n_disruption_couplings = 0
        self.n_disruption_oscillators = 0
        self.n_disruption_sensors = 0

        # Update object with provided keyword arguments
        # NOTE: This overrides the previous declarations
        self.__dict__.update(kwargs)
        print(self.duration)

