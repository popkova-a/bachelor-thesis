import chaospy
import sympy as sp
import numpy as np
from typing import Any, Iterable, Union


class DampedOscillator:
    def __init__(self, **kwargs):
        """
        Creates the damped linear oscillator model.

        :param kwargs:   dict, of the model parameters
                     :m:     float [kg],          body mass
                     :c:     sympy.Symbol,        damping factor (internal)
                     :k:     float [N/m],         spring constant
                     :f:     float [m],           forcing amplitude
                     :omega: float [rad/s],       cyclic frequency of the driving force
                     :y_0:   float [m],           initial position
                     :y_1:   float [m/s],         initial velocity
        """

        if len(kwargs) == 0:
            kwargs = {'m': 1, 'k': 0.035, 'f': 0.1, 'omega': 1, 'y_0': 0.5, 'y_1': 0}
        else:
            if not all(map(lambda x: isinstance(x, (int, float)), kwargs.values())):
                raise TypeError("Parameter values of the model must be of the int or float type.")

            if any(map(lambda x: np.isnan(x) or np.isinf(x), kwargs.values())):
                raise ValueError("All parameters of the system should be well-defined.")

        self.m = kwargs['m']
        self.c = sp.Symbol('c', real=True, positive=True)
        self.k = kwargs['k']
        self.f = kwargs['f']
        self.omega = kwargs['omega']
        self.y_0 = kwargs['y_0']
        self.y_1 = kwargs['y_1']
        self.displacement = None
        self.velocity = None

        if any(map(lambda x: x <= 0, [self.m, self.k, self.f, self.omega])):
            raise ValueError("Certain parameters of the system should be positive.")

    def solve(self) -> tuple:
        """
        Solves the damped linear oscillator model.

        :return: tuple, real parts of displacement and velocity
        """

        t = sp.Symbol('t', real=True, positive=True)
        y = sp.Function('y')
        eq = self.m * sp.diff(y(t), (t, 2)) + self.c * sp.diff(y(t), t) + self.k * y(t) - \
             self.f * sp.cos(self.omega * t)
        sol = sp.dsolve(eq, func=y(t), ics={y(0): self.y_0, sp.diff(y(t), t).subs(t, 0): self.y_1})

        self.displacement = sp.re(sol.rhs)
        self.velocity = sp.re(sp.diff(self.displacement, t))

        return self.displacement, self.velocity

    def lambdify_displacement(self, c: float = 0.1) -> callable:
        """
        Creates a Python function out of the displacement expression.

        :param c:  float,          value for the undetermined damping factor
        :return:   function,       for the displacement time dependence computation
        """

        if self.displacement is None:
            raise AttributeError("The displacement attribute wasn't found, solve the model first.")

        if not isinstance(c, (int, float)):
            raise TypeError("The damping factor must be of the int or float type.")

        if not(c >= 0):  # to catch NaN
            raise ValueError("The value of the damping factor should greater than or equal to zero.")

        t = sp.Symbol('t', real=True, positive=True)
        return sp.lambdify(t, self.displacement.subs(self.c, c, real=True), 'numpy')

    def lambdify_velocity(self, c: float = 0.1) -> callable:
        """
        Creates a Python function out of the velocity expression.

        :param c:  float,          value for the undetermined damping factor
        :return:   function,       for the velocity time dependence computation
        """

        if self.velocity is None:
            raise AttributeError("The velocity attribute wasn't found, solve the model first.")

        if not isinstance(c, (int, float)):
            raise TypeError("The damping factor must be of the int or float type.")

        if not(c >= 0):  # to catch NaN
            raise ValueError("The value of the damping factor should greater than or equal to zero.")

        t = sp.Symbol('t', real=True, positive=True)
        return sp.lambdify(t, self.velocity.subs(self.c, c, real=True), 'numpy')

    def compute_displacement(self, c: float = 0.1, coordinates: Iterable[float] = None) -> np.ndarray:
        """
        Computes the displacement values at the coordinate points using a certain
        value of the damping factor.

        :param c:            float,                 damping factor
        :param coordinates:  Iterable[float],       time points
        :return:             np.ndarray,            displacement values at the coordinates
        """

        if self.displacement is None:
            raise AttributeError("The displacement attribute wasn't found, solve the model first.")

        if not isinstance(c, (int, float)):
            raise TypeError("The damping factor must be of the int or float type.")

        if not(c >= 0):  # to catch NaN
            raise ValueError("The value of the damping factor should greater than or equal to zero.")

        if isinstance(coordinates, Iterable):
            try:
                coordinates = np.array(coordinates, dtype=float)
            except BaseException:
                raise TypeError("The coordinates must be of the iterable type over floats.")
        else:
            raise TypeError("The coordinates must be of the iterable type over floats.")

        if len(coordinates) == 0:
            raise ValueError("The input is empty.")

        if not all(map(lambda x: x >= 0, coordinates)):
            raise ValueError("The time points should be greater than or equal to zero.")

        displ_func = self.lambdify_displacement(c)

        return displ_func(coordinates)

    def compute_velocity(self, c: float = 0.1, coordinates: Iterable[float] = None) -> np.ndarray:
        """
        Computes the velocity values at the coordinate points using a certain
        value of the damping factor.

        :param c:            float,                 damping factor
        :param coordinates:  Iterable[float],       time points
        :return:             np.ndarray,            velocity values at the coordinates
        """

        if self.velocity is None:
            raise AttributeError("The velocity attribute wasn't found, solve the model first.")

        if not isinstance(c, (int, float)):
            raise TypeError("The damping factor must be of the int or float type.")

        if not(c >= 0):  # to catch NaN
            raise ValueError("The value of the damping factor should greater than or equal to zero.")

        if isinstance(coordinates, Iterable):
            try:
                coordinates = np.array(coordinates, dtype=float)
            except BaseException:
                raise TypeError("The coordinates must be of the iterable type over floats.")
        else:
            raise TypeError("The coordinates must be of the iterable type over floats.")

        if len(coordinates) == 0:
            raise ValueError("The input is empty.")

        if not all(map(lambda x: x >= 0, coordinates)):
            raise ValueError("The time points should be greater than or equal to zero.")

        vel_func = self.lambdify_velocity(c)

        return vel_func(coordinates)


def galerkin_ic(y_0: Union[int, float], v_0: Union[int, float], basis: Any) -> np.ndarray:
    """
    Computes the initial condition of the variables for the Galerkin method.

    :param y_0:    Union[int, float],       displacement initial value
    :param v_0:    Union[int, float],       velocity initial value
    :param basis:  Any,                     PC basis
    :return:       np.ndarray,              general initial conditions array
    """

    if not isinstance(y_0, (int, float)):
        raise TypeError("y_0 must be of the int ot float type.")

    if not isinstance(v_0, (int, float)):
        raise TypeError("v_0 must be of the int ot float type.")

    if np.isnan([y_0, v_0]).any() or np.isinf([y_0, v_0]).any():
        raise ValueError("The input contains NaN or Inf.")

    init_y = y_0 * np.eye(N=1, M=len(basis)).reshape(-1)
    init_v = v_0 * np.eye(N=1, M=len(basis)).reshape(-1)

    return np.hstack((init_y, init_v))


def galerkin_system(t: Union[int, float], x: np.ndarray, dist: Any, c_mu: Union[int, float],
                    c_sigma: Union[int, float], basis: Any, osc: DampedOscillator) -> np.ndarray:
    """
    Computes the system of ordinary linear explicit first order differential equations

    :param t:        Union[int, float],        time point
    :param x:        np.ndarray,               variables of the differential equation system
    :param dist:     Any,                      transformed distribution the PC basis is built upon
    :param c_mu:     Union[int, float],        mean of the untransformed initial distribution
    :param c_sigma:  Union[int, float],        standard deviation of the untransformed initial distribution
    :param basis:    Any,                      PC basis
    :param osc:      DampedOscillator,         damped oscillator model
    :return:         np.ndarray,               ODE system
    """

    if not isinstance(t, (int, float)):
        raise TypeError("The time point must be of the int ot float type.")

    if not(t >= 0):  # to catch NaN
        raise ValueError("The time point should greater than or equal to zero.")

    if len(x) != 2 * len(basis):
        raise ValueError("The array of variables must be twice longer than the PC basis.")

    if not(c_mu >= 0) or not(c_sigma >= 0):  # to catch NaN
        raise ValueError("The distribution parameters must be greater than or equal to zero.")

    if not isinstance(osc, DampedOscillator):
        raise TypeError("The damped oscillator model should be provided.")

    delta = np.eye(len(basis))
    y = x[:len(basis)]
    v = x[len(basis):]
    dydt = v
    dvdt = np.zeros(len(basis))
    for n in range(len(basis)):
        dvdt[n] = (osc.f * np.cos(osc.omega * t) * delta[0, n] - c_mu * v[n] - osc.k * y[n]) / osc.m
        summa = 0
        for i in range(len(basis)):
            summa += v[i] * chaospy.E(basis[1] * basis[i] * basis[n], dist)

        dvdt[n] += -c_sigma / (osc.m * np.sqrt(3)) * summa

    return np.hstack((dydt, dvdt))
