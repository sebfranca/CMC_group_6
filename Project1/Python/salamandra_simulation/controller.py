"""Network controller"""

import numpy as np
from farms_core.model.control import AnimatController


class SalamandraController(AnimatController):
    """Salamandra controller"""

    def __init__(self, joints_names, animat_data, network):
        super().__init__(
            joints_names=[joints_names, [], []],
            max_torques=[np.ones(len(joints_names)), [], []],
        )
        self.network = network
        self.animat_data = animat_data

    def step(self, iteration, time, timestep):
        """Control step"""
        # Lateral hydrodynamic forces
        loads = -1 * np.concatenate([
            self.animat_data.sensors.xfrc.array[
                iteration,
                0:self.network.robot_parameters['n_body_joints'],
                1
            ],
            np.zeros(self.network.robot_parameters['n_legs_joints'])
        ])
        # Network integration step
        self.network.step(iteration, time, timestep, loads)

    def positions(self, iteration, time, timestep):
        """Postions"""
        return dict(zip(
            self.joints_names[0],
            self.network.get_motor_position_output(iteration),
        ))

