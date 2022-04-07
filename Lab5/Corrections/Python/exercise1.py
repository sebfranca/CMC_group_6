""" Lab 5 - Exercise 1 """

import matplotlib.pyplot as plt
import numpy as np

import farms_pylog as pylog
from muscle import Muscle, MuscleResult
from mass import Mass
from cmcpack import DEFAULT, parse_args
from cmcpack.plot import save_figure
from system_parameters import MuscleParameters, MassParameters
from isometric_muscle_system import IsometricMuscleSystem
from isotonic_muscle_system import IsotonicMuscleSystem

from typing import Any
from nptyping import NDArray

DEFAULT["label"] = [r"$\theta$ [rad]", r"$d\theta/dt$ [rad/s]"]

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels

DEFAULT["save_figures"] = True


def exercise1a():
    """ Exercise 1a
    The goal of this exercise is to understand the relationship
    between muscle length and tension.
    Here you will re-create the isometric muscle contraction experiment.
    To do so, you will have to keep the muscle at a constant length and
    observe the force while stimulating the muscle at a constant activation."""

    # Defination of muscles
    parameters = MuscleParameters()
    pylog.warning("Loading default muscle parameters")
    pylog.info(parameters.showParameters())

    # Create muscle object
    muscle = Muscle(parameters)


    # Instatiate isometric muscle system
    sys = IsometricMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)

    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.l_opt # To get the muscle optimal length

    # Evalute for a single muscle stretch
    muscle_stretch: float = 0.2

    # Evalute for a single muscle stimulation
    muscle_stimulation: float = 1.

    # Set the initial condition
    x0: NDArray[(2,), float] = np.array([0.0, sys.muscle.l_opt])
    # x0[0] --> muscle stimulation intial value
    # x0[1] --> muscle contracticle length initial value

    # Set the time for integration
    t_start: float = 0.0
    t_stop: float = 0.2
    time_step: float = 0.001

    # shape is 1D. The length depends on paramaters
    time: NDArray[(Any,), float] = np.arange(t_start, t_stop, time_step)

    # Run the integration
    result: MuscleResult = sys.integrate(
        x0=x0,
        time=time,
        time_step=time_step,
        stimulation=muscle_stimulation,
        muscle_length=muscle_stretch
    )

    # Plotting
    plt.figure('Isometric muscle experiment')
    plt.plot(result.time, result.l_ce)
    plt.title('Isometric muscle experiment')
    plt.xlabel('Time [s]')
    plt.ylabel('Muscle contracticle length [m]')
    plt.grid()

    # Single Activation
    sys = IsometricMuscleSystem()
    sys.add_muscle(Muscle(MuscleParameters()))

    l_opt: float = sys.muscle.l_opt
    l_slack: float = sys.muscle.l_slack
    l_mtu: float = l_opt + l_slack
    muscle_stretch: NDArray[(Any,), float] = np.concatenate(
        (np.linspace(l_mtu - (l_mtu*0.5), l_mtu, 25),
         np.linspace(l_mtu, l_mtu + (l_mtu*0.5), 25)))

    length: NDArray[(Any,), float] = np.zeros(np.shape(muscle_stretch))
    active_force: NDArray[(Any,), float] = np.zeros(np.shape(muscle_stretch))
    passive_force: NDArray[(Any,), float] = np.zeros(np.shape(muscle_stretch))
    force: NDArray[(Any,), float] = np.zeros(np.shape(muscle_stretch))

    for idx, stretch_ in enumerate(muscle_stretch):
        result: MuscleResult = sys.integrate(
            x0=np.array([0, sys.muscle.l_opt]),
            time=np.arange(0, 0.2, 0.001),
            time_step=0.001,
            stimulation=1.,
            muscle_length=stretch_,
            )

        length[idx] = result.l_ce[-1]
        active_force[idx] = result.active_force[-1]
        passive_force[idx] = result.passive_force[-1]
        force[idx] = result.tendon_force[-1]

    # Plotting
    plt.figure('1a_Force-Length')
    plt.plot(length / sys.muscle.l_opt, force)
    plt.plot(length / sys.muscle.l_opt, active_force)
    plt.plot(length / sys.muscle.l_opt, passive_force)
    plt.plot(np.ones((25, 1)), np.linspace(0., sys.muscle.f_max, 25),
             '--')
    plt.text(0.6, 600, 'Below \n optimal \n length',
             fontsize=14)
    plt.text(1.1, 600, 'Above \n optimal \n length',
             fontsize=14)
    plt.title('Force-Length Relationship')
    plt.xlabel('Muscle CE Length')
    plt.ylabel('Muscle Force')
    plt.legend(('Force', 'Active Force', 'Passive Force'), loc=2)
    plt.grid()

    # Multiple Activation
    plt.figure('1b_Force-Length')
    l_opt: float = sys.muscle.l_opt
    l_slack: float = sys.muscle.l_slack
    l_mtu: float = l_opt + l_slack
    muscle_stretch: NDArray[(Any,), float] = np.linspace(l_mtu - (l_mtu*0.5),
                                 l_mtu + (l_mtu*0.5), 50)

    length: NDArray[(Any,), float] = np.zeros(np.shape(muscle_stretch))
    active_force: NDArray[(Any,), float] = np.zeros(np.shape(muscle_stretch))
    passive_force: NDArray[(Any,), float] = np.zeros(np.shape(muscle_stretch))
    force: NDArray[(Any,), float] = np.zeros(np.shape(muscle_stretch))

    for activation_ in np.arange(0.05, 1.0, 0.2):
        for idx, stretch_ in enumerate(muscle_stretch):
            result: MuscleResult = sys.integrate(x0=[0, sys.muscle.l_opt],
                                   time=np.arange(0, 0.2, 0.001),
                                   time_step=0.001,
                                   stimulation=activation_,
                                   muscle_length=stretch_)

            length[idx] = result.l_ce[-1]
            active_force[idx] = result.active_force[-1]
            passive_force[idx] = result.passive_force[-1]
            force[idx] = result.tendon_force[-1]
        # Plotting
        plt.plot(length / sys.muscle.l_opt, active_force)
    plt.plot(np.ones((25, 1)), np.linspace(0., sys.muscle.f_max, 25),
                 '--')
    plt.title('Force-Length Relationship for different Activations')
    plt.xlabel('Muscle CE Length')
    plt.ylabel('Muscle Force')
    plt.grid()
    plt.legend(
        tuple(
            ['Activation {:.2f}'.format(act)
                      for act in np.arange(0.05, 1.0, 0.2)
            ]))

    # # l_opt - 2.c
    l_opt: float = sys.muscle.l_opt - (sys.muscle.l_opt*0.25)
    sys.muscle.l_opt = l_opt
    l_slack: float = sys.muscle.l_slack
    l_mtu: float = l_opt + l_slack
    muscle_stretch: NDArray[(Any,), float] = np.concatenate(
        (np.linspace(l_mtu - (l_mtu*0.5), l_mtu, 25),
         np.linspace(l_mtu, l_mtu + (l_mtu*0.5), 25)))

    length: NDArray[(Any,), float]  = np.zeros(np.shape(muscle_stretch))
    active_force: NDArray[(Any,), float]  = np.zeros(np.shape(muscle_stretch))
    passive_force: NDArray[(Any,), float]  = np.zeros(np.shape(muscle_stretch))
    force: NDArray[(Any,), float]  = np.zeros(np.shape(muscle_stretch))

    for idx, stretch_ in enumerate(muscle_stretch):
        result: MuscleResult = sys.integrate(
            x0=np.array([0, sys.muscle.l_opt]),
            time=np.arange(0, 0.2, 0.001),
            time_step=0.001,
            stimulation=1.,
            muscle_length=stretch_,
        )

        length[idx] = result.l_ce[-1]
        active_force[idx] = result.active_force[-1]
        passive_force[idx] = result.passive_force[-1]
        force[idx] = result.tendon_force[-1]

    # Plotting
    plt.figure('1c_Short-Fiber-Length')
    plt.plot(length, force)
    plt.plot(length, active_force)
    plt.plot(length, passive_force)
    plt.plot(np.ones((25, 1))*sys.muscle.l_opt,
             np.linspace(0., sys.muscle.f_max, 25), '--')
    plt.title('Force-Length Relationship - Short Fiber length')
    plt.xlabel('Muscle CE Length')
    plt.ylabel('Muscle Force')
    plt.legend(('Force', 'Active Force', 'Passive Force'), loc='best')
    plt.grid()

    # # High Fiber Length
    l_opt: float = sys.muscle.l_opt + (sys.muscle.l_opt*0.25)
    sys.muscle.l_opt = l_opt
    l_slack: float = sys.muscle.l_slack
    l_mtu: float = l_opt + l_slack
    muscle_stretch: NDArray[(Any,), float]  = np.concatenate(
        (np.linspace(l_mtu - (l_mtu*0.5), l_mtu, 25),
         np.linspace(l_mtu, l_mtu + (l_mtu*0.5), 25)))

    length: NDArray[(Any,), float] = np.zeros(np.shape(muscle_stretch))
    active_force: NDArray[(Any,), float] = np.zeros(np.shape(muscle_stretch))
    passive_force: NDArray[(Any,), float] = np.zeros(np.shape(muscle_stretch))
    force: NDArray[(Any,), float] = np.zeros(np.shape(muscle_stretch))

    for idx, stretch_ in enumerate(muscle_stretch):
        result: MuscleResult = sys.integrate(
            x0=np.array([0, sys.muscle.l_opt]),
            time=np.arange(0, 0.2, 0.001),
            time_step=0.001,
            stimulation=1.,
            muscle_length=stretch_,
        )

        length[idx] = result.l_ce[-1]
        active_force[idx] = result.active_force[-1]
        passive_force[idx] = result.passive_force[-1]
        force[idx] = result.tendon_force[-1]
    # # Plotting
    plt.figure('1c_Long-Fiber-Length')
    plt.plot(length, force)
    plt.plot(length, active_force)
    plt.plot(length, passive_force)
    plt.plot(np.ones((25, 1))*sys.muscle.l_opt,
             np.linspace(0., sys.muscle.f_max, 25), '-')
    plt.title('Force-Length Relationship - Long Fiber length')
    plt.xlabel('Muscle CE Length')
    plt.ylabel('Muscle Force')
    plt.legend(('Force', 'Active Force', 'Passive Force'), loc='best')
    plt.grid()



def exercise1d():
    """ Exercise 1d

    Under isotonic conditions external load is kept constant.
    A constant stimulation is applied and then suddenly the muscle
    is allowed contract. The instantaneous velocity at which the muscle
    contracts is of our interest."""

    # Defination of muscles
    muscle_parameters: MuscleParameters= MuscleParameters()
    print(muscle_parameters.showParameters())

    mass_parameters: MassParameters = MassParameters()
    print(mass_parameters.showParameters())

    # Create muscle object
    muscle = Muscle(muscle_parameters)

    # Create mass object
    mass = Mass(mass_parameters)


    # Instatiate isotonic muscle system
    sys = IsotonicMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)

    # Add the mass to the system
    sys.add_mass(mass)

    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.l_opt # To get the muscle optimal length

    # Evalute for a single load
    load: float = 250/9.81

    # Evalute for a single muscle stimulation
    muscle_stimulation: float = 1.0

    # Set the initial condition
    x0: NDArray[(4,), float] = np.array([
        0.0,
        sys.muscle.l_opt,
        sys.muscle.l_opt + sys.muscle.l_slack,
        0.0,
    ])
    # x0[0] - -> activation
    # x0[1] - -> contractile length(l_ce)
    # x0[2] - -> position of the mass/load
    # x0[3] - -> velocity of the mass/load

    # Set the time for integration
    t_start: float = 0.0
    t_stop: float = 0.4
    time_step: float = 0.001
    time_stabilize: float = 0.2

    time: NDArray[(Any,), float] = np.arange(t_start, t_stop, time_step)

    # Run the integration
    result: MuscleResult = sys.integrate(
        x0=x0,
        time=time,
        time_step=time_step,
        time_stabilize=time_stabilize,
        stimulation=muscle_stimulation,
        load=load
    )

    # Plotting
    plt.figure('Isotonic muscle experiment')
    plt.plot(result.time,
             result.v_ce)
    plt.title('Isotonic muscle experiment')
    plt.xlabel('Time [s]')
    plt.ylabel('Muscle contracticle velocity [lopts/s]')
    plt.grid()

    # Single Activation

    load: NDArray[(Any,), float] = np.arange(1., 350., 10.)

    sys = IsotonicMuscleSystem()
    sys.add_muscle(Muscle(MuscleParameters()))
    sys.add_mass(Mass(MassParameters()))
    v_ce: NDArray[(Any,), float] = np.zeros(np.shape(load))
    force: NDArray[(Any,), float] = np.zeros(np.shape(load))
    x0: NDArray[(4,), float] = np.array([
        0.0,
        sys.muscle.l_opt,
        sys.muscle.l_opt + sys.muscle.l_slack,
        0.0
    ])

    for idx, _load in enumerate(load):
        result: MuscleResult = sys.integrate(
            x0=x0,
            time=np.arange(0, 0.3, 0.001),
            time_step=0.001,
            time_stabilize=0.2,
            stimulation=1,
            load=_load
        )
        force[idx] = result.tendon_force[-1]
        if(result.l_mtu[-1] > sys.muscle.l_opt + sys.muscle.l_slack):
            v_ce[idx] = max(result.v_ce[int(0.2/0.001):-1])
        else:
            v_ce[idx] = min(result.v_ce[int(0.2/0.001):-1])
    plt.figure('1d_Force-Velocity')
    plt.plot(v_ce, force / sys.muscle.f_max, '*')
    plt.plot(v_ce, force / sys.muscle.f_max)
    plt.plot(0.0, 1.0, 'ro', markersize='10')
    plt.annotate('Isometric contraction',
                 xy=(0.0, 1.0), xycoords='data',
                 xytext=(0.8, 0.95), textcoords='axes fraction',
                 arrowprops=dict(
                     arrowstyle='->',
                 ),
                 horizontalalignment='right', verticalalignment='top')
    plt.plot(np.zeros((25, 1)),
             np.linspace(0., 2., 25), '--r')
    plt.text(-0.6, 0.55, 'Eccentric \n contraction',
             fontsize=12)
    plt.arrow(
        0.0, 1.0, -0.2, 0.0, head_width=0.05, head_length=0.05
    )
    plt.text(0.2, 0.55, 'Concentric \n contraction',
             fontsize=12)
    plt.arrow(
        0.0, 1.0, 0.2, 0.0, head_width=0.05, head_length=0.05
    )
    plt.title('Force-Velocity Relationship')
    plt.xlabel('Normalized CE velocity')
    plt.ylabel('Normalized tension')

    # Multiple Activation
    plt.figure('1f_Force-Velocity')
    load: NDArray[(Any,), float] = np.arange(1., 350., 10.)

    for activation_ in np.arange(0.1, 1., 0.2):
        force = np.zeros(np.shape(load))
        v_ce = np.zeros(np.shape(load))
        for idx, _load in enumerate(load):
            result: MuscleResult = sys.integrate(
                x0=[
                    0.0,
                    sys.muscle.l_opt,
                    sys.muscle.l_opt + sys.muscle.l_slack,
                    0.0,
                ],
                time=np.arange(0, 0.3, 0.001),
                time_step=0.001,
                time_stabilize=0.2,
                stimulation=activation_,
                load=_load,
            )
            force[idx] = result.tendon_force[-1]
            if(result.l_mtu[-1] > sys.muscle.l_opt + sys.muscle.l_slack):
                v_ce[idx] = max(result.v_ce[int(0.2/0.001):-1])
            else:
                v_ce[idx] = min(result.v_ce[int(0.2/0.001):-1])
        plt.plot(v_ce, force / sys.muscle.f_max,'*')
        plt.plot(
            v_ce, force / sys.muscle.f_max,
            label='Activation {:.2f}'.format(activation_)
        )
    plt.title('Force-Velocity Relationship for multiple activations')
    plt.xlabel('Normalized CE velocity')
    plt.ylabel('Normalized external load')
    plt.grid()
    plt.legend()
    # plt.legend(tuple(['Activation {}'.format(act)
    #                   for act in np.arange(0.05, 1.0, 0.2)]))



def exercise1():
    exercise1a()
    exercise1d()

    if DEFAULT["save_figures"] is False:
        plt.show()
    else:
        figures = plt.get_figlabels()
        print(figures)
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
            plt.close(fig)


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    exercise1()