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
        self.feedback_gains = np.zeros(self.n_oscillators)
        self.exercise_8b = False
        
        
        self.turn = None
        self.drive_offset_turn = parameters.drive_offset_turn

        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        self.exercise_8b = parameters.exercise_8b
        self.turn = parameters.turn
        
        
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # phi_ij
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i
        self.set_feedback_gains(parameters)  # K_fb

    def set_frequencies(self, parameters):
        """Set frequencies"""
        limbSaturatesLow = lambda x: x<1
        limbSaturatesHigh = lambda x: x>3
        bodySaturatesLow = lambda x: x<1
        bodySaturatesHigh = lambda x: x>5
        f_drive_body = lambda x: 0.2*x + 0.3
        f_drive_limb = lambda x: 0.2*x
        
        freqs = np.zeros(20)
        
        d = parameters.drive_mlr
        
        left =  [0,1,2,3,4,5,6,7,16,18]
        right = [8,9,10,11,12,13,14,15,17,19]
        
        #Turning modifications
        if self.turn is None:
            d_r = d
            d_l = d
        elif self.turn == "right":
            d_r = d + self.drive_offset_turn
            d_l = d - self.drive_offset_turn
        elif self.turn == "left":
            d_r = d - self.drive_offset_turn
            d_l = d + self.drive_offset_turn
        
            
        for i in range(16):
            if i in left and not bodySaturatesHigh(d_l) and not bodySaturatesLow(d_l):
                    freqs[i] = f_drive_body(d_l)
            elif i in right and not bodySaturatesHigh(d_r) and not bodySaturatesLow(d_r):
                    freqs[i] = f_drive_body(d_r)
        for i in range(16,20):
            if i in left and not limbSaturatesHigh(d_l) and not limbSaturatesLow(d_l):
                    freqs[i] = f_drive_limb(d_l)
            elif i in right and not limbSaturatesHigh(d_r) and not limbSaturatesLow(d_r):
                    freqs[i] = f_drive_limb(d_r)
                   
            
            
        self.freqs = freqs
        #print(freqs[16:])

    def set_coupling_weights(self, parameters):
        """Set coupling weights"""
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
            phase_lag_params = {
            'b2b_same' : [-2*np.pi/8],
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
        """Set nominal amplitudes"""
        if self.exercise_8b:
            self.nominal_amplitudes = parameters.nominal_amplitudes
            
        else:
            d = parameters.drive_mlr
            nominal_amplitudes = np.zeros(20)
            
            left =  [0,1,2,3,4,5,6,7,16,18]
            right = [8,9,10,11,12,13,14,15,17,19]
            
            #Turning modifications
            if self.turn is None:
                d_r = d
                d_l = d
            elif self.turn == "right":
                d_r = d + self.drive_offset_turn
                d_l = d - self.drive_offset_turn
            elif self.turn == "left":
                d_r = d - self.drive_offset_turn
                d_l = d + self.drive_offset_turn
            
                
            limbSaturatesLow = lambda x: x<1
            limbSaturatesHigh = lambda x: x>3
            bodySaturatesLow = lambda x: x<1
            bodySaturatesHigh = lambda x: x>5
            r_drive_body = lambda x: 0.065*x + 0.196
            r_drive_limb = lambda x: 0.131*x + 0.131
            
            
            for i in range(16):
                if i in left and not bodySaturatesHigh(d_l) and not bodySaturatesLow(d_l):
                        nominal_amplitudes[i] = r_drive_body(d_l)
                elif i in right and not bodySaturatesHigh(d_r) and not bodySaturatesLow(d_r):
                        nominal_amplitudes[i] = r_drive_body(d_r)
            for i in range(16,20):
                if i in left and not limbSaturatesHigh(d_l) and not limbSaturatesLow(d_l):
                        nominal_amplitudes[i] = r_drive_limb(d_l)
                elif i in right and not limbSaturatesHigh(d_r) and not limbSaturatesLow(d_r):
                        nominal_amplitudes[i] = r_drive_limb(d_r)
            
            
        
            self.nominal_amplitudes = nominal_amplitudes
        #print(nominal_amplitudes[16:])

    def set_feedback_gains(self, parameters):
        """Set feeback gains"""
        pylog.warning('Convergence rates must be set')


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
                        
        
        
        # for i in range(20):
        #     for j in range(20):
        #         if i==j: pass
                
        #         elif isBody(i) and isBody(j):
        #             if bodyOnSameSide(i, j) and j==i+1:
        #                 matrix[i,j] = b2b_same
        #                 if couplingM:
        #                     matrix[j,i] = b2b_same
        #                 else:
        #                     matrix[j,i] = - b2b_same
        #             elif bodyOnOppSide(i, j) and abs(i-j)==8:
        #                 matrix[i,j] = b2b_opp
                        
        #         elif isLimb(i) and isLimb(j):
        #             if limbOnSameSide(i, j):
        #                 matrix[i,j] = l2l_same
        #             elif limbOnOppSide(i, j) and (abs(i-j)==2 or (abs(i-j)==1 and not(i==17 and j==18) and not(i==18 and j==17))):
        #                 matrix[i,j] = l2l_opp
                
        #         elif isBody(i) and isLimb(j):
        #             if bodyLimbOnSameSide(i, j):
        #                 if (j in frontLimbs and i in frontBodies) or (j in backLimbs and i in backBodies):
        #                     matrix[j,i] = l2b

        return matrix
