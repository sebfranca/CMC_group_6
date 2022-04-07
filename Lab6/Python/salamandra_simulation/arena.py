"""Arena"""

import os
from scipy.spatial.transform import Rotation
from farms_core.model.options import ArenaOptions
from farms_core.simulation.options import SimulationOptions
from farms_bullet.model.model import SimulationModels, DescriptionFormatModel


def get_arena(
        arena_options: ArenaOptions,
        simulation_options: SimulationOptions,
) -> SimulationModels:
    """Get arena from options"""

    # Options
    meters = simulation_options.units.meters
    orientation = Rotation.from_euler(
        seq='xyz',
        angles=arena_options.orientation,
        degrees=False,
    ).as_quat()

    # Main arena
    arena = DescriptionFormatModel(
        path=arena_options.sdf,
        spawn_options={
            'posObj': [pos*meters for pos in arena_options.position],
            'ornObj': orientation,
        },
        load_options={'units': simulation_options.units},
    )

    # Ground
    if arena_options.ground_height is not None:
        arena.spawn_options['posObj'][2] += (
            arena_options.ground_height*meters
        )

    # Water
    if arena_options.water.height is not None:
        assert os.path.isfile(arena_options.water.sdf), (
            'Must provide a proper sdf file for water:'
            f'\n{arena_options.water.sdf} is not a file'
        )
        arena = SimulationModels(models=[
            arena,
            DescriptionFormatModel(
                path=arena_options.water.sdf,
                spawn_options={
                    'posObj': [0, 0, arena_options.water.height*meters],
                    'ornObj': [0, 0, 0, 1],
                },
                load_options={'units': simulation_options.units},
            ),
        ])

    return arena

