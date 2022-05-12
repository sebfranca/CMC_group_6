"""Simulation parameters"""

import numpy as np


class SimulationParameters:
    """Simulation parameters"""

    def __init__(self, **kwargs):
        super(SimulationParameters, self).__init__()
        # Default parameters
        self.n_body_joints = 8
        self.n_legs_joints = 4
        self.duration = 30
        self.initial_phases = None
        self.phase_lag_body = None
        self.amplitude_gradient = None
        
        self.drive_mlr = kwargs.get("drive_mlr",2) #default = 2
        self.exercise_8b = kwargs.get("exercise_8b",False)
        self.turn = kwargs.get("turn",None)
        self.drive_offset_turn = 1
        
        self.nominal_amplitudes = kwargs.get("nominal_amplitudes",np.zeros(20))
        self.phase_bias = kwargs.get("phase_bias",np.zeros([20,20]))
        
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

