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
        self.drive = 1
        self.cf1_body = 0.2
        self.cf2_body = 0.3
        self.cf1_limb = 0.2
        self.cf2_limb = 0
        self.cr1_body = 0.065
        self.cr2_body = 0.196
        self.cr1_limb = 0.131
        self.cr2_limb = 0.131
        self.threshold_l_body = 1
        self.threshold_h_body = 5
        self.threshold_l_limb = 1
        self.threshold_h_limb = 3
        self.iteration = 0
        
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
        self.set_drive(parameters)
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # theta_i
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i
        
        
    def set_drive(self,parameters):
        if hasattr(parameters, 'drive'):
            self.drive = parameters.drive

    def set_frequencies(self, parameters):
        """Set frequencies"""
        if self.iteration == 0:
            if hasattr(parameters,'freqs'):
                self.freqs = parameters.freqs
            else:
                self.freqs = np.zeros(20)
        
        elif not(isinstance(self.drive,int) or isinstance(self.drive,float)):
            #BODY
            if self.drive[self.iteration] < self.threshold_l_body or self.drive[self.iteration] > self.threshold_h_body:
                self.freqs[0:self.n_oscillators_body] = np.zeros_like(self.freqs[0:self.n_oscillators_body]) #saturates
            else:
                self.freqs[0:self.n_oscillators_body] = np.ones_like(self.freqs[0:self.n_oscillators_body]) * (self.cf1_body*self.drive[self.iteration] + self.cf2_body)
            
            #LIMBS
            if self.drive[self.iteration] < self.threshold_l_limb or self.drive[self.iteration] > self.threshold_h_limb:
                self.freqs[self.n_oscillators_body:] = np.zeros_like(self.freqs[self.n_oscillators_body:])
            else:
                self.freqs[self.n_oscillators_body:] = np.ones_like(self.freqs[self.n_oscillators_body:]) * (self.cf1_limb*self.drive[self.iteration] + self.cf2_limb)

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
        if self.iteration == 0:
            if hasattr(parameters,'nominal_amplitudes'):
                self.nominal_amplitudes = parameters.nominal_amplitudes
            else:
                self.nominal_amplitudes = np.ones(20)
                
        elif not(isinstance(self.drive,int) or isinstance(self.drive,float)):
            if self.drive[self.iteration] < self.threshold_l_body or self.drive[self.iteration] > self.threshold_h_body:
                self.nominal_amplitudes[0:self.n_oscillators_body] = np.zeros_like(self.nominal_amplitudes[0:self.n_oscillators_body]) #saturates
            else:
                self.nominal_amplitudes[0:self.n_oscillators_body] = np.ones_like(self.nominal_amplitudes[0:self.n_oscillators_body]) * (self.cr1_body*self.drive[self.iteration] + self.cr2_body)
            
            if self.drive[self.iteration] < self.threshold_l_limb or self.drive[self.iteration] > self.threshold_h_limb:
                self.nominal_amplitudes[self.n_oscillators_body:] = np.zeros_like(self.nominal_amplitudes[self.n_oscillators_body:])
            else:
                self.nominal_amplitudes[self.n_oscillators_body:] = np.ones_like(self.nominal_amplitudes[self.n_oscillators_body:]) * (self.cr1_limb*self.drive[self.iteration] + self.cr2_limb)