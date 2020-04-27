import numpy as np
from scipy.optimize import newton
from scipy.integrate import solve_ivp
from diff_equations import implicit_eigen


def implicit(V, xlim, n):
    hbar = 1
    m = 1

    return implicit_eigen(lambda x: -hbar ** 2 / (2 * m), lambda x: 0, V, xlim, n)


def shooting(V, xlim, n, initial):
    hbar = 1
    m = 1

    def func(_x, _psi, _E):
        u = np.zeros(2)
        u[0] = _psi[1]
        u[1] = -2 * m / hbar ** 2 * (_E - V(_x)) * _psi[0]
        return u

    def get_boundary(_E, _x):
        sol = solve_ivp(func, xlim, initial, args=[_E], t_eval=_x)
        return sol.y[0][-1]

    x = np.linspace(*xlim, n)
    E = newton(get_boundary, 0, args=[x])
    sol = solve_ivp(func, xlim, initial, args=[E], t_eval=x)
    return E, sol.t, sol.y[0]
