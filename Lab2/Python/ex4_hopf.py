""" Hopf oscillator """

from nptyping import NDArray

# pylint: disable=invalid-name


class HopfParameters:
    """ Hopf Parameters """

    def __init__(self, mu, omega):
        super().__init__()
        self.mu: float = mu
        self.omega: float = omega

    def __str__(self) -> str:
        return self.msg()

    def msg(self) -> str:
        """ Message """
        return f'mu: {self.mu}, omega: {self.omega}'

    def check(self):
        """ Check parameters """
        assert self.mu >= 0, 'Mu must be positive'
        assert self.omega >= 0, 'Omega must be positive'


class CoupledHopfParameters(HopfParameters):
    """ Coupled Hopf Parameters """

    def __init__(self, mu, omega, k):
        super().__init__(mu, omega)
        self.k: float = k

    def msg(self) -> str:
        """ Message """
        return f'mu: {self.mu}, omega: {self.omega}, k: {self.k}'

    def check(self):
        """ Check parameters """
        assert self.mu >= 0, 'Mu must be positive'
        assert self.omega >= 0, 'Omega must be positive'
        assert self.k >= 0, 'K must be positive'


def hopf_equation(
        x: NDArray[(2,), float],
        _time: float = None,
        params: HopfParameters = HopfParameters(mu=1., omega=1.0),
) -> NDArray[(2,), float]:
    """ Hopf oscillator equation """
    mu = params.mu
    omega = params.omega
    # biolog.warning('Hopf oscillator equation must be implemented')
    return [0, 0]


def coupled_hopf_equation(
        x: NDArray[(4,), float],
        _time: float = None,
        params: HopfParameters = None,
) -> NDArray[(4,), float]:
    """ Coupled Hopf oscillator equation """
    if params is None:
        params = CoupledHopfParameters(
            mu=[1., 1.],
            omega=[1.0, 1.2],
            k=[-0.5, -0.5]
        )
    mu = params.mu
    omega = params.omega
    k = params.k
    # biolog.warning('Coupled Hopf oscillator equation must be implemented')
    return [0, 0, 0, 0]

