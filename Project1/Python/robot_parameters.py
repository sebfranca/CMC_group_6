"""Robot parameters"""

import numpy as np
from farms_core import pylog
from simulation_parameters import SimulationParameters #used for turning


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super(RobotParameters, self).__init__()

        # Default parameters needed for the network
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.initial_phases = parameters.initial_phases
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
        self.rates = 20*np.ones(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)
        self.drive_mlr = 0
        
        #Parameters used to force the definition of matrices, instead
        #of using the default values or the drive equation
        self.exercise_8b = False
        self.exercise_8c = False
        self.decoupled = False
        self.exercise_8f = False
        self.nominal_amplitude_parameters = np.zeros(2) #Rhead and Rtail
                
        #Parameters used for turning and backward motion
        self.timestep = 0 #used in network.py::network_ode
        self.backward = False
        self.drive_offset_turn = 0
        self.isturning = False
        self.turns = ["None" for i in range(1)]
        
        #Parameters used to modulate CPG and feedback
        self.cpg_active = True
        self.fb_active  = False
        self.feedback_gains = np.zeros(self.n_oscillators)
        
        #Disruption parameters
        self.n_disruption_couplings = 0
        self.n_disruption_oscillators = 0
        self.n_disruption_sensors = 0
        self.couplings_remaining = [i for i in range(15) if i!=7]
        self.oscillators_remaining = [i for i in range(16)]
        self.disrupted_oscillators = []
        self.sensors_remaining = [i for i in range(16)]
        self.disrupted_sensors = []
        
        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        #Directly from the parameters
        self.exercise_8b = parameters.exercise_8b
        self.exercise_8c = parameters.exercise_8c
        self.decoupled = parameters.decoupled
        self.exercise_8f = parameters.exercise_8f
        self.backward = parameters.backward
        self.drive_mlr = parameters.drive_mlr
        self.timestep = parameters.timestep
        self.turns = parameters.turns
        self.drive_offset_turn = parameters.drive_offset_turn
        self.cpg_active = parameters.cpg_active
        self.fb_active = parameters.fb_active
        self.nominal_amplitude_parameters = parameters.nominal_amplitude_parameters
        self.n_disruption_couplings = parameters.n_disruption_couplings
        self.n_disruption_oscillators = parameters.n_disruption_oscillators
        self.n_disruption_sensors = parameters.n_disruption_sensors
        
        #Based on some functions
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # phi_ij
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i
        self.set_feedback_gains(parameters)  # K_fb
        
        #apply disruptions
        while self.n_disruption_couplings >0:
            self.disrupt_couplings()
            self.n_disruption_couplings -= 1
            
        while self.n_disruption_oscillators>0:
            self.disrupt_oscillators()
            self.n_disruption_oscillators -= 1
            
        while self.n_disruption_sensors >0:
            self.disrupt_sensors()
            self.n_disruption_sensors -= 1

    def perform_turn(self,instruction):    
        """Update the frequencies and amplitudes during turning, 
        by changing the drive."""
        turn_parameters = SimulationParameters(
            turn_instruction = instruction,
            )
        
        self.set_frequencies(turn_parameters)
        self.set_nominal_amplitudes(turn_parameters)
        self.isturning = True
        
    def end_turn(self):
        """Reset the frequencies and amplitudes in straight line,
        by resetting the drive."""
        straight_parameters = SimulationParameters(
            turn_instruction = "None")
        self.set_frequencies(straight_parameters)
        self.set_nominal_amplitudes(straight_parameters)
        self.isturning = False

    def disrupt_couplings(self):
        target = np.random.choice(self.couplings_remaining)
        self.couplings_remaining = [c for c in self.couplings_remaining if c!=target]
        
        self.coupling_weights[target,target+1] = 0
        self.coupling_weights[target+1,target] = 0
        
    def disrupt_oscillators(self):
        target = np.random.choice(self.oscillators_remaining)
        self.oscillators_remaining = [o for o in self.oscillators_remaining if o!=target]
        self.disrupted_oscillators.append(target)

    def disrupt_sensors(self):
        target = np.random.choice(self.sensors_remaining)
        self.sensors_remaining = [s for s in self.sensors_remaining if s!=target]
        self.disrupted_sensors.append(target)

    def set_frequencies(self, parameters):
        """Set frequencies
        --> left and right oscillators may receive a different drive,
        if the salamandra must turn left or right.
        Steps:
            1. Define the saturation functions and f(drive) according to 
            the paper
            2. Update the drive if necessary, during turns
            3. Update the frequencies
        """
        limbSaturatesLow = lambda x: x<1
        limbSaturatesHigh = lambda x: x>3
        bodySaturatesLow = lambda x: x<1
        bodySaturatesHigh = lambda x: x>5
        
        f_drive_body = lambda x: 0.2*x + 0.3
        f_drive_limb = lambda x: 0.2*x
        
        left =  [0,1,2,3,4,5,6,7,16,18]
        right = [8,9,10,11,12,13,14,15,17,19]
        
        d = self.drive_mlr        
        #Turning modifications
        if parameters.turn_instruction == "right":
            self.d_r = d - self.drive_offset_turn
            self.d_l = d + self.drive_offset_turn
        elif parameters.turn_instruction == "left":
            self.d_r = d + self.drive_offset_turn
            self.d_l = d - self.drive_offset_turn
        else:
            self.d_r = d
            self.d_l = d
        
        freqs = np.zeros(20)
        for i in range(16):
            if i in left and not bodySaturatesHigh(self.d_l) and not bodySaturatesLow(self.d_l):
                    freqs[i] = f_drive_body(self.d_l)
            elif i in right and not bodySaturatesHigh(self.d_r) and not bodySaturatesLow(self.d_r):
                    freqs[i] = f_drive_body(self.d_r)
        for i in range(16,20):
            if i in left and not limbSaturatesHigh(self.d_l) and not limbSaturatesLow(self.d_l):
                    freqs[i] = f_drive_limb(self.d_l)
            elif i in right and not limbSaturatesHigh(self.d_r) and not limbSaturatesLow(self.d_r):
                    freqs[i] = f_drive_limb(self.d_r)
                   
        self.freqs = freqs

    
        
    def set_coupling_weights(self, parameters):
        """Set coupling weights"""
        if self.decoupled:
            coupling_params = {
                'b2b_same' : [0],
                'b2b_opp' : [10],
                'l2l_same' : [10],
                'l2l_opp' : [10],
                'l2b' : [30]
                }        
        elif self.exercise_8f:
            coupling_params = {
                'b2b_same' : [parameters.b2b_same_coupling],
                'b2b_opp' : [10],
                'l2l_same' : [10],
                'l2l_opp' : [10],
                'l2b' : [30]
                }    
        else:
            coupling_params = {
                'b2b_same' : [10],
                'b2b_opp' : [10],
                'l2l_same' : [10],
                'l2l_opp' : [10],
                'l2b' : [30]
                }        
        self.coupling_weights = self.make_matrix(coupling_params)
        
    def set_phase_bias(self, parameters):
        """Set phase bias"""
        if self.exercise_8b:
            self.phase_bias = parameters.phase_bias
        else:
            if self.backward:
                #if going backward, invert the direction of coupling
                #by putting a minus sign in b2b_same
                phase_lag_params = {
                'b2b_same' : [-2*np.pi/8],
                'b2b_opp' : [np.pi],
                'l2l_same' : [np.pi],
                'l2l_opp' : [np.pi],
                'l2b' : [0]
                }
            else:
                phase_lag_params = {
                'b2b_same' : [2*np.pi/8],
                'b2b_opp' : [np.pi],
                'l2l_same' : [np.pi],
                'l2l_opp' : [np.pi],
                'l2b' : [0]
                }
        
            self.phase_bias = self.make_matrix(phase_lag_params, couplingM=False)

    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        self.rates = 20*np.ones(self.n_oscillators)

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes
        --> left and right oscillators may receive a different drive,
        if the salamandra must turn left or right.
        Steps:
            1. Define the saturation functions and f(drive) according to 
            the paper
            2. Update the drive if necessary, during turns
            3. Update the amplitudes
            
        In 8b and 8c, this algorithm is bypassed."""
        limbSaturatesLow = lambda x: x<1
        limbSaturatesHigh = lambda x: x>3
        bodySaturatesLow = lambda x: x<1
        bodySaturatesHigh = lambda x: x>5.
        
        r_drive_body = lambda x: 0.065*x + 0.196
        r_drive_limb = lambda x: 0.131*x + 0.131
        
        left =  [0,1,2,3,4,5,6,7,16,18]
        right = [8,9,10,11,12,13,14,15,17,19]
        
        d = self.drive_mlr
        
        nominal_amplitudes = np.zeros(20)
        if self.exercise_8b:
            self.nominal_amplitudes = parameters.nominal_amplitudes
        
        elif self.exercise_8c:
            Rhead = self.nominal_amplitude_parameters[0]
            Rtail = self.nominal_amplitude_parameters[1]
            
            adjusted_r_drive_body = np.linspace(Rhead, Rtail, 8)
            adjusted_r_drive_body = np.append(adjusted_r_drive_body, adjusted_r_drive_body)
        
            for i in range(16):
                if not bodySaturatesHigh(d) and not bodySaturatesLow(d):
                    nominal_amplitudes[i] = adjusted_r_drive_body[i]
            for i in range(16,20):
                if not limbSaturatesHigh(d) and not limbSaturatesLow(d):
                    nominal_amplitudes[i] = r_drive_limb(d)

            self.nominal_amplitudes = nominal_amplitudes
        
        else:
            #Turning modifications
            if parameters.turn_instruction == "right":
                self.d_r = d - self.drive_offset_turn
                self.d_l = d + self.drive_offset_turn
            elif parameters.turn_instruction == "left":
                self.d_r = d + self.drive_offset_turn
                self.d_l = d - self.drive_offset_turn
            else:
                self.d_r = d
                self.d_l = d
                
            for i in range(16):
                if i in left and not bodySaturatesHigh(self.d_l) and not bodySaturatesLow(self.d_l):
                        nominal_amplitudes[i] = r_drive_body(self.d_l)
                elif i in right and not bodySaturatesHigh(self.d_r) and not bodySaturatesLow(self.d_r):
                        nominal_amplitudes[i] = r_drive_body(self.d_r)
            for i in range(16,20):
                if i in left and not limbSaturatesHigh(self.d_l) and not limbSaturatesLow(self.d_l):
                        nominal_amplitudes[i] = r_drive_limb(self.d_l)
                elif i in right and not limbSaturatesHigh(self.d_r) and not limbSaturatesLow(self.d_r):
                        nominal_amplitudes[i] = r_drive_limb(self.d_r)
            
            self.nominal_amplitudes = nominal_amplitudes

    def set_feedback_gains(self, parameters):
        """Set feeback gains"""        
        self.feedback_gains = parameters.fb_gain * np.concatenate(([-1]*8, [1]*8, [-1,1]*2))


    def make_matrix(self, params, i=0, couplingM=True):
        """
        Creates the coupling or phase bias matrix.
        
        Parameters
        ----------------
        *coupling or phase lag values:
            b2b_ : body to body
            l2l_ : limb to limb
            l2b : limb to body
            
            _same : on the same side (left or right)
            _opp : opposite sides (left or right)
            
        *i: not used here (always =0), but used in exercise_8b.py
        It gives the index in the grid
        
        *couplingM: True if making the coupling matrix,
        False if making the phase bias matrix.
        """

        b2b_same = params['b2b_same'][i]
        b2b_opp = params['b2b_opp'][i]
        l2l_same = params['l2l_same'][i]
        l2l_opp = params['l2l_opp'][i]
        l2b = params['l2b'][i]
        
        #All conditions used in the matrix definition
        #(used for clarity)
        isLimb = lambda j: j in [16,17,18,19]
        isBody = lambda i: i in [k for k in range(16)]
            
        limbOnLeft = lambda i: i%2 ==0
        limbOnRight = lambda i: i%2==1
        bodyOnLeft = lambda j: j in [k for k in range(8)]
        bodyOnRight = lambda j: j in [k for k in range(8,16)]
        
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
        
        
        matrix = np.zeros([20,20])
        for i in range(20):
            if isBody(i):
                j = i+1
                if bodyOnSameSide(i, j):
                    matrix[i,j] = b2b_same
                    
                    if couplingM: matrix[j,i] =  b2b_same
                    else:         matrix[j,i] = -b2b_same
                
                j = i+8
                if bodyOnOppSide(i, j):
                    matrix[i,j] = b2b_opp
                    matrix[j,i] = b2b_opp
                
            elif isLimb(i):
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
