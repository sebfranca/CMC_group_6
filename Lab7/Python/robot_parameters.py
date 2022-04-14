"""Robot parameters"""

import numpy as np
from farms_core import pylog


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super(RobotParameters, self).__init__()

        # Initialise parameters
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.n_joints = self.n_body_joints + self.n_legs_joints
        self.n_oscillators_body = 2*self.n_body_joints
        self.n_oscillators_legs = self.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs
        
        self.freqs = np.zeros(self.n_oscillators)
        self.coupling_weights = np.zeros([
            self.n_oscillators,
            self.n_oscillators,
        ])
        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators])
        self.rates = np.zeros(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)
        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # theta_i
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i

    def set_frequencies(self, parameters):
        """Set frequencies"""
        if hasattr(parameters, 'freqs'):
            self.freqs = parameters.freqs

    def set_coupling_weights(self, parameters):
        """Set coupling weights"""
        if hasattr(parameters, 'coupling_weights'):
            self.coupling_weights = parameters.coupling_weights

    def set_phase_bias(self, parameters):
        """Set phase bias"""
        if hasattr(parameters,'phase_bias'):
            self.phase_bias = parameters.phase_bias

    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        if hasattr(parameters,'rates'):
            self.rates = parameters.rates

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
        if hasattr(parameters,'nominal_amplitudes'):
            self.nominal_amplitudes = parameters.nominal_amplitudes