""" Lab 2 """

from typing import Any

import numpy as np
from nptyping import NDArray
import matplotlib.pyplot as plt

import farms_pylog as pylog
from cmcpack import integrate, DEFAULT, parse_args
from cmcpack.plot import bioplot, save_figure

from ex3_pendulum import PendulumParameters, pendulum_system
from ex3_analyse import fixed_points  # Optional


DEFAULT['label'] = [r'$\theta$ [rad]', r'$d\theta/dt$ [rad/s]']


def pendulum_perturbation(
        state: NDArray[(2,), float],
        time: float = None,
        parameters: PendulumParameters = None,
) -> NDArray[(2,), float]:
    """ Function for system integration with perturbation """
    torque = 0.0  # : Default external torque set to zero
    #: torque perturbation
    if 5 < time < 5.1:
        torque = 10.0
    #: state perturbation
    if 10 < time < 10.1:
        state[0], state[1] = 0.5, 0.0
    return pendulum_system([state[0], state[1]], time, parameters, torque)


def evolution_cases(time: NDArray[(Any,), float]):
    """ Normal simulation """
    pylog.info('Evolution with basic paramater')
    x0_cases = [
        ['Normal', [0.1, 0]],
        ['Stable', [0.0, 0.0]],
        ['Unstable', [np.pi, 0.0]],
        ['Multiple loops', [0.1, 10.0]]
    ]
    title = '{} case {} (x0={})'
    parameters = PendulumParameters()
    for name, x_0 in x0_cases:
        res = integrate(pendulum_system, x_0, time, args=(parameters,))
        res.plot_state(title.format(name, 'state', x_0))
        res.plot_phase(title.format(name, 'phase', x_0))


def fixed_point_types(time: NDArray[(Any,), float]):
    """Fixed points types

    Critically damped and overdamped cases for the fixed point in (0,0) for
    d**2 >= 4*g/L

    """

    pylog.info('Evolution with modified parameters')
    parameters = PendulumParameters()

    figname = 'Fixed point types'
    labs = ['Underdamped', 'Critically damped', 'Overdamped']

    param_cases = [0.5, 1.0, 1.5]
    initial_state = [np.pi/2, 0.0]
    states = []
    for pfact in param_cases:
        parameters.d = pfact * 2 * np.sqrt(parameters.g/parameters.L)
        res = integrate(
            ode_fun=pendulum_system,
            x0=initial_state,
            time=time,
            args=(parameters,),
        )
        states.append(res.state)
    states = np.array(states)

    bioplot(
        data_x=states[:, :, 0].T,
        data_y=time,
        figure=figname + '_Temporal evolution',
        label=labs,
    )

    plt.figure(figname + '_Phase plot')
    for i, lab in enumerate(labs):
        plt.plot(states[i, :, 0], states[i, :, 1], linewidth=2.0, label=lab)
    plt.xlabel(r'$\theta$ [rad]')
    plt.ylabel(r'$d\theta/dt$ [rad/s]')
    plt.legend()
    plt.grid('True')
    save_figure(figname + '_Phase plot')


def evolution_no_damping(
        x_0: NDArray[(2,), float],
        time: NDArray[(Any,), float],
):
    """ No damping simulation """
    pylog.info('Evolution with no damping')
    parameters = PendulumParameters(d=0.0)
    pylog.info(parameters)
    title = '{} without damping (x0={})'
    res = integrate(pendulum_system, x_0, time, args=(parameters,))
    res.plot_state(title.format('State', x_0))
    res.plot_phase(title.format('Phase', x_0))


def evolution_perturbation(
        x_0: NDArray[(2,), float],
        time: NDArray[(Any,), float],
):
    """ Perturbation and no damping simulation """
    pylog.info('Evolution with perturbations')
    parameters = PendulumParameters(d=0.0)
    pylog.info(parameters)
    title = '{} with perturbation (x0={})'
    res = integrate(
        pendulum_perturbation, x_0, time, args=(parameters,)
    )
    res.plot_state(title.format('State', x_0))
    res.plot_phase(title.format('Phase', x_0))


def evolution_dry(x_0, time):
    """ Dry friction simulation """
    pylog.info('Evolution with dry friction')
    parameters = PendulumParameters(d=0.03, dry=True)
    pylog.info(parameters)
    title = '{} with dry friction (x0={})'
    res = integrate(pendulum_system, x_0, time, args=(parameters,))
    res.plot_state(title.format('State', x_0))
    res.plot_phase(title.format('Phase', x_0))


def exercise3(clargs):
    """ Exercise 3 """
    fixed_points()  # Optional
    parameters = PendulumParameters()  # Checkout pendulum.py for more info
    pylog.info(parameters)
    # Simulation parameters
    time = np.arange(0, 30, 0.01)  # Simulation time
    x_0 = [0.1, 0.0]  # Initial state


    # Evolutions
    evolution_cases(time)
    x_0 = [0.1, 0]
    evolution_no_damping(x_0, time)
    time1 = np.arange(0, 20, 0.01)  # Simulation time
    fixed_point_types(time1)
    evolution_perturbation(x_0, time)
    evolution_dry(x_0, time)

    # Show plots of all results
    if not clargs.save_figures:
        plt.show()


if __name__ == '__main__':
    CLARGS = parse_args()
    exercise3(CLARGS)