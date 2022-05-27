"""Oscillator network ODE"""

import numpy as np
from scipy.integrate import ode
from robot_parameters import RobotParameters


def network_ode(_time, state, robot_parameters, loads):
    """Network_ODE

    Parameters
    ----------
    _time: <float>
        Time
    state: <np.array>
        ODE states at time _time
    robot_parameters: <RobotParameters>
        Instance of RobotParameters

    Returns
    -------
    :<np.array>
        Returns derivative of state (phases and amplitudes)

    """
    # Implementation of turning:
        #get the iteration and the current turn instruction
        #update robot_parameters (freqs + amplitudes) accordingly:
            #turn if needed --> change the drive
            #end turn if needed --> reset the drive
    iteration = round(_time/robot_parameters.timestep)  
    if robot_parameters.turns[iteration] != "None":
        instruction = robot_parameters.turns[iteration]
        robot_parameters.perform_turn(instruction)
    else:
        if robot_parameters.isturning:
            robot_parameters.end_turn()
            
    
    #get the states and parameters
    n_oscillators = robot_parameters.n_oscillators
    phases = state[:n_oscillators]
    amplitudes = state[n_oscillators:2*n_oscillators]
    
    a = robot_parameters.rates
    R = robot_parameters.nominal_amplitudes
    
    
    thetadot = 2*np.pi*robot_parameters.freqs 
    
    #CPG implementation
    if robot_parameters.cpg_active:
        #avoid double loop --> about 5x faster!
        #first, make a square matrix with phases
        theta_i = np.resize(np.repeat(phases,n_oscillators),[n_oscillators,n_oscillators]) 
        theta_j = theta_i.T
        
        sin_matrix = np.sin(theta_j - theta_i - robot_parameters.phase_bias.T).T
        #Implementation of eq. (3) without loops
        #result is correct only if we transpose again then we take the diagonal
        thetadot += np.diagonal(
                        np.dot(
                            amplitudes* robot_parameters.coupling_weights.T,
                            sin_matrix))
        
        for osc in range(8):
            if osc in robot_parameters.disrupted_oscillators:
                thetadot[osc] = 0
                thetadot[osc+8] = 0
        
    #Sensory feedback implementation
    if robot_parameters.fb_active:
        #12 bodies but 20 oscillators --> loads must be reshaped
        loads = np.concatenate((
            loads[0:8], loads[0:8],
            loads[8:]
            ))
        
        loads = [0 if L in robot_parameters.disrupted_sensors
                 else L
                 for L in range(20)]
        
        thetadot += np.multiply(np.multiply(robot_parameters.feedback_gains, loads), np.cos(phases))
    
    
    
    rdot = a * (R-amplitudes)
        
    return np.concatenate([thetadot, rdot])


def motor_output(phases, amplitudes, iteration):
    """Motor output

    Parameters
    ----------
    phases: <np.array>
        Phases of the oscillator
    amplitudes: <np.array>
        Amplitudes of the oscillator

    Returns
    -------
    : <np.array>
        Motor outputs for joint in the system.

    """
    q = np.zeros_like(phases)[:12] + np.zeros_like(amplitudes)[:12]
    
    
    for i,r in enumerate(amplitudes):
        if i<8:
            q[i] = r*(1+np.cos(phases[i])) - amplitudes[i+8]*(1+np.cos(phases[i+8]))
        elif i<12:
            #/!\ 20 phases but only 12 bodies --> i+8
            q[i] = 1*phases[i+8] - np.pi/2
    return q



class SalamandraNetwork:
    """Salamandra oscillator network"""

    def __init__(self, sim_parameters, n_iterations, state):
        super().__init__()
        self.n_iterations = n_iterations
        # States
        self.state = state
        # Parameters
        self.robot_parameters = RobotParameters(sim_parameters)
        # Set initial state
        if sim_parameters.set_seed:
            np.random.seed(sim_parameters.randseed)
        self.state.set_phases(
            iteration=0,
            value=1e-3*np.random.rand(self.robot_parameters.n_oscillators),
        )
        # Set solver
        self.solver = ode(f=network_ode)
        self.solver.set_integrator('dopri5')
        self.solver.set_initial_value(y=self.state.array[0], t=0.0)

    def step(self, iteration, time, timestep, loads=None):
        """Step"""
        if loads is None:
            loads = np.zeros(self.robot_parameters.n_joints)
        if iteration + 1 >= self.n_iterations:
            return
        self.solver.set_f_params(self.robot_parameters, loads)
        self.state.array[iteration+1, :] = self.solver.integrate(time+timestep)

    def outputs(self, iteration=None):
        """Oscillator outputs"""
        
        phases = self.state.phases(iteration=iteration)
        amplitudes = self.state.amplitudes(iteration=iteration)
        
        return amplitudes*(1+np.cos(phases))

    def get_motor_position_output(self, iteration=None):
        """Get motor position"""
        return motor_output(
            self.state.phases(iteration=iteration),
            self.state.amplitudes(iteration=iteration),
            iteration=iteration,
        )

