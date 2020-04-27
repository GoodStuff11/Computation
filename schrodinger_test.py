import schrodinger as s
import matplotlib.pyplot as plt
from numpy import sin, cos, exp, pi
import numpy as np
import sys

if __name__ == '__main__':
    a = 1
    initial = [0, 0.4]
    xlim = [0, a]
    n = 200

    np.set_printoptions(threshold=sys.maxsize)


    def V(x):
        # try:
        #     return [30*i if i < a / 2 else 30 for i in x]
        # except:
        #     if x < a / 2:
        #         return 0
        #     return 30
        return 0

    plt.figure("Shooting")

    plt.grid()
    E, x, psi = s.shooting(V, xlim, n, initial)
    plt.plot(x, psi)
    # plt.plot(x, V(x))
    # plt.hlines([E, E], *xlim)
    plt.xlabel('x')
    plt.ylabel('$\Psi(x)$')
    plt.title('E={}'.format(E))

    plt.figure("Implicit")
    plt.grid()
    E_list, x, psi_list = s.implicit(V, xlim, n)

    plt.plot(x, psi_list[0])
    # plt.plot(x, V(x))
    # plt.hlines([E, E], *xlim)
    plt.xlabel('x')
    plt.ylabel('$\Psi(x)$')
    plt.title('E={}'.format(E_list[0]))
    plt.show()

    # exact: E=4.93480220054=pi^2/2
