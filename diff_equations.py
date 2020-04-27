import numpy as np
import matplotlib.pyplot as plt


def implicit_solve(a, b, c, d, xlim, ylim, n):
    """
    a(x) y'' + b(x) y' + c(x) y = d(x)
    y(x0) = y0
    y(xf) = yf

    (a(x) is redundant but this is a proof of concept)
    :param a:
    :param b:
    :param c:
    :param xlim: tuple with two float elements
    :param ylim: tuple with two float elements
    :param n:
    :return:
    """
    x0, xn = xlim
    y0, yn = ylim

    h = (xn - x0) / n

    A = np.zeros([n - 2, n - 2])
    B = np.zeros([n - 2, n - 2])
    C = np.zeros([n - 2, n - 2])
    D = np.zeros(n - 2)

    diff2 = np.zeros([n - 2, n - 2])
    diff1 = np.zeros([n - 2, n - 2])
    for i in range(n - 2):  # i + 1 goes from 1 to n-1
        diff2[min(i + 1, n - 3), i] = 1  # n - 1
        diff2[i, min(i + 1, n - 3)] = 1  # n + 1
        diff2[i, i] = -2  # n

        diff1[min(i + 1, n - 3), i] = 1 / 2  # n - 1
        diff1[i, min(i + 1, n - 3)] = 3 / 2  # n + 1
        diff1[i, i] = -2  # n

        A[i, i] = a(x0 + h * (i + 1))
        B[i, i] = b(x0 + h * (i + 1))
        C[i, i] = c(x0 + h * (i + 1))
        D[i] = d(x0 + h * (i + 1))

    # initial and final conditions
    D[0] -= diff2[1, 0] * A[0, 0] * y0 / h ** 2 + \
            diff1[1, 0] * B[0, 0] * y0 / h
    D[-1] -= diff2[0, 1] * A[-1, -1] * yn / h ** 2 + \
             diff1[0, 1] * B[-1, -1] * yn / h

    y_middle = np.linalg.solve(
        np.matmul(A, diff2) +
        np.matmul(B, diff1) * h +
        C * h ** 2,
        D * h ** 2)
    y = np.array([y0] + list(y_middle) + [yn])
    x = np.linspace(x0, xn, n, endpoint=True)
    return x, y


def implicit_eigen(a, b, c, xlim, n):
    """
    a(x) y'' + b(x) y' + c(x) y = lambda y

    ylim is (0,0)
    :param a:
    :param b:
    :param xlim:
    :param n:
    :return:
    """
    x0, xn = xlim

    h = (xn - x0) / n

    A = np.zeros([n - 2, n - 2])
    B = np.zeros([n - 2, n - 2])
    C = np.zeros([n - 2, n - 2])

    diff2 = np.zeros([n - 2, n - 2])
    diff1 = np.zeros([n - 2, n - 2])
    for i in range(n - 2):  # i + 1 goes from 1 to n-1
        diff2[min(i + 1, n - 3), i] = 1  # n - 1
        diff2[i, min(i + 1, n - 3)] = 1  # n + 1
        diff2[i, i] = -2  # n

        diff1[min(i + 1, n - 3), i] = 1 / 2  # n - 1
        diff1[i, min(i + 1, n - 3)] = 3 / 2  # n + 1
        diff1[i, i] = -2  # n

        A[i, i] = a(x0 + h * (i + 1))
        B[i, i] = b(x0 + h * (i + 1))
        C[i, i] = c(x0 + h * (i + 1))

    lam_list, sol_list = np.linalg.eigh(np.matmul(A, diff2) +
                                        np.matmul(B, diff1) * h +
                                        C * h ** 2)

    return lam_list / h ** 2, \
           np.linspace(x0, xn, n, endpoint=True), \
           np.array([[0] + list(sol_list[:, i]) + [0] for i in range(n - 2)])


if __name__ == '__main__':
    n = 200
    N = n + 1

    xlim = (0, 10)
    ylim = (0, 0)

    a = lambda x: -1
    b = lambda x: 0
    c = lambda x: np.exp(x)
    d = lambda x: x

    # sol = implicit_solve(a, b, c, d, xlim, ylim, n)
    E_list, x, y_list = implicit_eigen(a, b, c, xlim, n)

    for i in range(n):
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(x, y_list[i])
        plt.title('E={}'.format(E_list[i]))
        plt.show()
