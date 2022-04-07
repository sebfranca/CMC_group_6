""" This contains the methods to simulate isometric muscle contraction. """

import numpy as np
import matplotlib.pyplot as plt
from cmcpack.results import Result

import farms_pylog as pylog
from cmcpack import integrate

from muscle import Muscle, MuscleResult
from system_parameters import MuscleParameters

from typing import Any
from nptyping import NDArray


class IsometricMuscleSystem(object):
    """System to simulate isometric muscle system.
    Results
    -------
        iso_sys: <IsometricMuscleSystem>
            Instance of IsometricMuscleSystem
    """

    def __init__(self):
        """ Initialization.
        Parameters
        ----------
        None

        Example:
        -------
        >>> isometric_system = IsometricMuscleSystem()
        """
        super(IsometricMuscleSystem, self).__init__()
        self.muscle = None

    def add_muscle(self, muscle: Muscle):
        """Add the muscle model to the system.

        Parameters
        ----------
        muscle: <Muscle>
            Instance of muscle model

        Example:
        --------
        >>> from muscle import Muscle
        >>> from system_parameters import MuscleParameters
        >>> muscle = Muscle(MuscleParameters()) #: Default muscle
        >>> isometric_system = IsometricMuscleSystem()
        >>> isometric_system.add_muscle(muscle)
        """
        if self.muscle != None:
            pylog.warning(
                'You have already added the muscle model to the system.')
            return
        else:
            if muscle.__class__ is not Muscle:
                pylog.error(
                    'Trying to set of type {} to muscle'.format(muscle.__class__))
                raise TypeError()
            else:
                pylog.info('Added new muscle model to the system')
                self.muscle = muscle

    def integrate(
        self,
        x0: NDArray[(2,), float],
        time: NDArray[(Any,), float],
        time_step: float = None,
        stimulation: float = 1.0,
        muscle_length: float = None,
    ) -> MuscleResult:
        """ Method to integrate the muscle model.

        Parameters:
        ----------
            x0 : <array>
                Initial state of the muscle
                    x0[0] --> activation
                    x0[1] --> contractile length (l_ce)
            time : <array>
                Time vector
            time_step : <float>
                Time step to integrate (Good value is 0.001)
            stimulation : <float>
                Muscle stimulation
            muscle_length : <float>
                Muscle length/stretch for isometric condition


        Returns:
        --------
            result : <MuscleResult>
            result.time :
                Time vector
            result.activation :
                Muscle activation state
            result.l_ce :
                Length of contractile element
            result.v_ce :
                Velocity of contractile element
            result.l_mtu :
                Total muscle tendon length
            result.active_force :
                 Muscle active force
            result.passive_force :
                Muscle passive force
            result.tendon_force :
                Muscle tendon force

        Example:
        --------
            >>> import nump as np
            >>> from muscle import Muscle
            >>> from system_parameters import MuscleParameters
            >>> muscle = Muscle(MuscleParameters()) #: Default muscle
            >>> isometric_system = IsometricMuscleSystem()
            >>> isometric_system.add_muscle(muscle)
            >>> # Initial state
            >>> x0 = [0, isometric_system.muscle.l_opt]
            >>> time_step = 0.001
            >>> t_start = 0.0
            >>> t_stop = 0.2
            >>> #: Time
            >>> time = np.arange(t_start, t_stop, time_step)
            >>> # Args take stimulation and muscle_length as input
            >>> # Set the muscle length to which you want to evaluate
            >>> muscle_stretch = 0.25
            >>> # Set the muscle stimulation to which you want to evaluate
            >>> muscle_stimulation = 0.5
            >>> args = (muscle_stimulation, muscle_stretch)
            >>> result = isometric_system.integrate(x0, time, time_step, args)
            >>> # results contain the states and the internal muscle
            >>> # attributes neccessary to complete the exercises

        The above example shows how to run the isometric condition once.
        In the exercise1.py file you have to use this setup to loop
        over multiple muscle stretch/muscle stimulation values to answer
        the questions.
        """

        #: If time step is not provided then compute it from the time vector
        if time_step == None:
            time_step = time[1] - time[0]

        #: If muscle length is missing
        if muscle_length == None:
            pylog.warning("Muscle length not provided, using default length")
            muscle_length = self.muscle.l_opt + self.muscle.l_slack

        #: Integration
        #: Instatiate the muscle results container
        self.muscle.instantiate_result_from_state(time)

        #: Run step integration
        for idx, _time in enumerate(time):
            res = self.step(
                x0, [_time, _time+time_step],
                stimulation, muscle_length)
            x0 = res.state[-1][:]
            self.muscle.generate_result_from_state(idx, _time,
                                                   muscle_length,
                                                   res.state[-1][:])
        return self.muscle.Result

    def step(self, x0, time, *args):
        """ Step the system."""
        res = integrate(self.muscle.dxdt, x0, time, args=args)
        return res



def main():
    """Main"""

    muscle_stretch = np.concatenate(
        (np.linspace(0.7, .9, 25), np.linspace(0.9, 1.3, 25)))

    sys = IsometricMuscleSystem()
    sys.add_muscle(Muscle(MuscleParameters()))
    length = np.empty(np.shape(muscle_stretch))
    active_force = np.empty(np.shape(muscle_stretch))
    passive_force = np.empty(np.shape(muscle_stretch))
    force = np.empty(np.shape(muscle_stretch))
    for idx, stretch_ in enumerate(muscle_stretch):
        result = sys.integrate(x0=[0, 0.4],
                               time=np.arange(0, 0.2, 0.001),
                               time_step=0.001,
                               stimulation=1.,
                               muscle_length=stretch_)
        length[idx] = result.l_mtu[-1]
        active_force[idx] = result.active_force[-1]
        passive_force[idx] = result.passive_force[-1]
        force[idx] = result.tendon_force[-1]

    l_opt = sys.muscle.l_opt
    plt.plot(np.array(length) / l_opt, force)
    plt.plot(np.array(length)/l_opt, active_force)
    plt.plot(np.array(length) / l_opt, passive_force)
    plt.grid()
    plt.xlabel('l_ce[m]')
    plt.ylabel('force[N]')
    plt.show()


if __name__ == '__main__':
    main()