""" Lab 2 - Exercise 4 """

import numpy as np
import matplotlib.pyplot as plt

import farms_pylog as pylog
from cmcpack import integrate, integrate_multiple, parse_args

from ex4_hopf import (
    HopfParameters,
    CoupledHopfParameters,
    hopf_equation,
    coupled_hopf_equation
)


def hopf_ocillator():
    """ 4a - Hopf oscillator simulation """
    pylog.warning('Hopf oscillator must be implemented')
    # 1. Set desired parameters for hopf oscillator by replacing
    #         < > with actual values
    #
    # params = HopfParameters(
    #     mu=<mu>,
    #     omega=<omega>,
    # )

    # 2. Create array of time
    #
    # time = np.arange(<start time>, <end time>, <time step>)

    # 3. Set title of the plot
    #
    # title = 'Hopf oscillator {} (x0={})'

    # 4. Set the label of the two variables of hopf oscillator
    #
    # label = ['x0', 'x1']

    # 5. Create a list of multiple initial values of the oscillator
    #    initial value should be provided as a list of 2 variables
    #    (initial condition = [x0, x1]) where x0 and x1 are float values.
    #
    # x0_list = [
    #     <initial condition 1>, eg. [0.5, 0.5]
    #     <initial condition 2>, eg. [1.0, 1.0]
    #     < .. >,
    # ]

    # 6. Plot state evolution for two variables for different initial
    #    values using the for loop
    #
    # for x_0 in x0_list:
    #     6.1 Use integrate function to integrate predefined hopf_equation with
    #         parameters defined above where:
    #         - x_0: initial value of oscillator taken from the list of values
    #         - time: array of time for integration
    #         - args=(params,) where params is an object of HopfParameters

    #     hopf = integrate(
    #         hopf_equation,
    #         x_0,
    #         time,
    #         args=(params,)
    #     )

    #     6.2 Plot integrated values of the hopf variables
    #         Here the plot_state() plots the state of hopf variables

    #     hopf.plot_state(title.format('state', x_0), label)

    # 7. Use the integrate_multiple() function to integrate hopf oscillator
    #    for multiple initial values together (without for loop) to later
    #    take advantage of inbuilt plot_phase function to visualize the
    #    evolution of different initial values in phase plot

    # 7.1 integrate predefined hopf_equation with parameters define above
    #     where:
    #     - x0_list: list of initial value conditions
    #     - time: array of time for integration
    #     - args=(params,) where params is an object of HopfParameters
    #
    # hopf_multiple = integrate_multiple(
    #     hopf_equation,
    #     x0_list,
    #     time,
    #     args=(params,)
    # )

    # 7.2 Plot phase plot of hopf variables with different initial conditions
    #
    # hopf_multiple.plot_phase(title.format('phase', x0_list), label=label)


def coupled_hopf_ocillator():
    """ 4b - Coupled Hopf oscillator simulation """
    pylog.warning('Coupled Hopf oscillator must be implemented')
    # 1. Set desired parameters for coupled hopf oscillator by replacing
    #         < > with actual values
    #
    # param = CoupledHopfParameters(
    #     mu=[<mu1>, <mu2>],
    #     omega=[<omega1>, <omega2>],
    #     k=[<k1>, <k2>]
    # )

    # 2. Create array of time
    #
    # time = np.arange(<start time>, <end time>, <time step>)

    # 3. Integrate predefined coupled_hopf_equation with parameters define
    #    above where:
    #      - x0: initial value condition for coupled hopf oscillator variables
    #      - time: array of time for integration
    #      - args=(params,): where params is an object of CoupledHopfParameters
    #
    # hopf = integrate(coupled_hopf_equation, x0, time, args=(param,))

    # 4. Generate plots of coupled hopf oscillator use the plot_state and
    #    plot_angle but first, some set the value required by the plots

    # 4.1 Set title
    # title = 'Coupled Hopf oscillator {} (x0={}, {})'

    # 4.2 set the labels of coupled_hopf_ocillator variables
    # label = ['x0', 'x1', 'x2', 'x3']

    # 4.3 set labels for angles of coupled_hopf_oscillator
    # label_angle = ['angle0', 'angle1']

    # 4.4 plot states
    # hopf.plot_state(title.format('state', x0, param), label, n_subs=2)

    # 4.5 plot angles
    # hopf.plot_angle(title.format('angle', x0, param), label_angle)


def exercise4(clargs):
    """ Exercise 4 """
    hopf_ocillator()
    coupled_hopf_ocillator()
    # Show plots of all results
    if not clargs.save_figures:
        plt.show()


if __name__ == '__main__':
    CLARGS = parse_args()
    exercise4(CLARGS)

