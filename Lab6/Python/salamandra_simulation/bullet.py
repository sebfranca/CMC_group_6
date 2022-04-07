"""Bullet"""

import pybullet

from farms_core.units import SimulationUnitScaling
from farms_core.simulation.options import SimulationOptions
from farms_core.model.options import AnimatOptions, ArenaOptions
from farms_bullet.model.model import SimulationModels
from farms_bullet.simulation.simulation import AnimatSimulation
from farms_bullet.swimming.drag import SwimmingHandler
from farms_bullet.model.animat import Animat
from farms_bullet.interface.interface import (
    Interfaces,
    UserParameters,
    DebugParameter,
)
from .arena import get_arena
from .controller import SalamandraController


class SalamandraUserParameters(UserParameters):
    """Salamandra user parameters"""

    def __init__(self, options):
        super().__init__(options=options)
        self['drive_left'] = DebugParameter('Drive left', 0, 0, 6)
        self['drive_right'] = DebugParameter('Drive right', 0, 0, 6)

    def drive_left(self):
        """Drive left"""
        return self['drive_left']

    def drive_right(self):
        """Drive right"""
        return self['drive_right']


class Salamandra(Animat):
    """Salamandra animat"""

    def __init__(
            self,
            sdf: str,
            options: AnimatOptions,
            controller: SalamandraController,
            timestep: float,
            iterations: int,
            units: SimulationUnitScaling,
    ):
        super().__init__(
            options=options,
            data=controller.animat_data if controller is not None else None,
            units=units,
        )
        self.sdf = sdf
        self.timestep = timestep
        self.n_iterations = iterations
        self.controller = controller
        self.xfrc_plot = None

    def spawn(self):
        """Spawn amphibious"""
        super().spawn()

        # Links masses
        link_mass_multiplier = {
            link.name: link.mass_multiplier
            for link in self.options.morphology.links
        }
        for link, index in self.links_map.items():
            if link in link_mass_multiplier:
                mass, _, torque, *_ = pybullet.getDynamicsInfo(
                    bodyUniqueId=self.identity(),
                    linkIndex=index,
                )
                pybullet.changeDynamics(
                    bodyUniqueId=self.identity(),
                    linkIndex=index,
                    mass=link_mass_multiplier[link]*mass,
                )
                pybullet.changeDynamics(
                    bodyUniqueId=self.identity(),
                    linkIndex=index,
                    localInertiaDiagonal=link_mass_multiplier[link]*torque,
                )

        # Debug
        self.xfrc_plot = [
            [
                False,
                pybullet.addUserDebugLine(
                    lineFromXYZ=[0, 0, 0],
                    lineToXYZ=[0, 0, 0],
                    lineColorRGB=[0, 0, 0],
                    lineWidth=3*self.units.meters,
                    lifeTime=0,
                    parentObjectUniqueId=self.identity(),
                    parentLinkIndex=i
                )
            ]
            for i in range(self.data.sensors.xfrc.array.shape[1])
        ] if self.options.show_xfrc else []


class SalamandraPybulletSimulation(AnimatSimulation):
    """Salamandra simulation"""

    def __init__(
            self,
            animat: Animat,
            simulation_options: SimulationOptions,
            arena_options: ArenaOptions = None,
    ):
        if arena_options is not None:
            arena = get_arena(arena_options, simulation_options)
        super().__init__(
            models=SimulationModels(
                [animat, arena]
                if arena_options is not None
                else [animat]
            ),
            options=simulation_options,
            interface=Interfaces(
                user_params=SalamandraUserParameters(
                    options=simulation_options,
                )
            )
        )

        # Swimming handling
        self.swimming_handler: SwimmingHandler = (
            SwimmingHandler(animat)
            if animat.options.physics.drag
            or animat.options.physics.sph
            else None
        )
        if isinstance(self.swimming_handler, SwimmingHandler):
            self.swimming_handler.set_xfrc_scale(
                animat.options.scale_xfrc
            )

    def update_controller(self, iteration: int):
        """Update controller"""
        self.animat().controller.step(
            iteration=iteration,
            time=iteration*self.options.timestep,
            timestep=self.options.timestep,
        )

    def step(self, iteration: int):
        """Simulation step"""
        animat = self.animat()

        # Interface
        if not self.options.headless:
            self.animat_interface(iteration)

        # Animat sensors
        animat.sensors.update(iteration)

        # Swimming
        if self.swimming_handler is not None:
            self.swimming_handler.step(iteration)

        # Update animat controller
        if animat.controller is not None:
            self.update_controller(iteration)

    def animat_interface(self, iteration: int):
        """Animat interface"""
        animat = self.animat()

        # Left
        if self.interface.user_params.drive_left().changed:
            animat.data.network.drives.array[iteration, 0] = (
                self.interface.user_params.drive_left().value
            )
            self.interface.user_params.drive_left().changed = False

        # Right
        if self.interface.user_params.drive_right().changed:
            animat.data.network.drives.array[iteration, 1] = (
                self.interface.user_params.drive_right().value
            )
            self.interface.user_params.drive_right().changed = False

