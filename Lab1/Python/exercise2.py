""" Lab 1 - Exercise 2 """

from typing import Any
import numpy as np
from nptyping import NDArray
import matplotlib.pyplot as plt

import farms_pylog as pylog
from cmcpack import integrate, parse_args, plot

# pylint: disable=invalid-name


def ode(
        x: NDArray[(2,), float],
        _: float = None,
        A: NDArray[(2, 2), float] = np.eye(2)
) -> NDArray[(2,), float]:
    """ System x_dot = A*x """
    return np.dot(A, x)


def integration(
        x0: NDArray[(2,), float],
        time: NDArray[(Any,), float],
        A: NDArray[(2, 2), float],
        name: str,
        **kwargs,
):
    """ System integration """
    labels = kwargs.pop('label', [f'State {i}' for i in range(2)])
    sys_int = integrate(ode, x0, time, args=(A,))
    sys_int.plot_state(f'{name}_state', labels)
    sys_int.plot_phase(f'{name}_phase')


def exercise2(clargs):
    """ Exercise 2 """
    pylog.info('Running exercise 2')

    # System definition
    pylog.warning('Proper matrix A must be implemented')
    A = np.array([[1, 0], [0, 1]])
    time_total = 10
    time_step = 0.01
    x0, time = [0, 1], np.arange(0, time_total, time_step)

    # Normal run
    pylog.warning('System integration must be implemented')
    # integration(x0, time, A, 'example')

    # Stable point (Optional)
    pylog.warning('Stable point integration must be implemented')

    # Periodic
    pylog.warning('Periodic system must be implemented')

    # Saddle
    pylog.warning('Saddle node system must be implemented')

    # Plot
    if not clargs.save_figures:
        plt.show()


if __name__ == '__main__':
    CLARGS = parse_args()
    exercise2(CLARGS)

