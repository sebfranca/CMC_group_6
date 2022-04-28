"""Python controller"""

from farms_core import pylog
from exercise_example import exercise_example
from exercise_8b import exercise_8b
from exercise_8c import exercise_8c
from exercise_8d import exercise_8d1, exercise_8d2
from exercise_8e import exercise_8e1, exercise_8e2
from exercise_8f import exercise_8f
from exercise_8g import exercise_8g1, exercise_8g2, exercise_8g3


def exercise_all(arguments):
    """Run all exercises"""

    verbose = 'not_verbose' not in arguments

    if not verbose:
        pylog.set_level('warning')

    # Timestep
    timestep = 1e-2

    # Exercise example to show how to run a grid search
    if 'exercise_example' in arguments:
        exercise_example(timestep)

    # Exercise 8b - Phase lag + amplitude study
    if '8b' in arguments:
        exercise_8b(timestep)

    # Exercise 8c - Amplitude gradient
    if '8c' in arguments:
        exercise_8c(timestep)

    # Exercise 8d1 - Turning
    if '8d1' in arguments:
        exercise_8d1(timestep)

    # Exercise 8d2 - Backwards swimming
    if '8d2' in arguments:
        exercise_8d2(timestep)

    # Exercise 8e1 - Uncoupled network
    if '8e1' in arguments:
        exercise_8e1(timestep)

    # Exercise 8e2 - Sensory feedback + Uncoupled network
    if '8e2' in arguments:
        exercise_8e2(timestep)

    # Exercise 8f - Sensory feedback + CPG
    if '8f' in arguments:
        exercise_8f(timestep)

    # Exercise 8g1 - Robustness of CPG only
    if '8g1' in arguments:
        exercise_8g1(timestep)

    # Exercise 8g2 - Robustness of Feedback only
    if '8g2' in arguments:
        exercise_8g2(timestep)

    # Exercise 8g3 - Robustness of the combined network
    if '8g3' in arguments:
        exercise_8g3(timestep)

    if not verbose:
        pylog.set_level('debug')


if __name__ == '__main__':
    exercises = [
        '8b',
        '8c',
        '8d1',
        '8d2',
        '8e1',
        '8e2',
        '8f',
        '8g1',
        '8g2',
        '8g3',
    ]
    exercise_all(arguments=exercises)

