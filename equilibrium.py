from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import time


def eqn_to_odes(reactions, k_values):
    """
    Converts system of reactions into a system of differential
    equations which can be read by solve_ivp

    reactions[0][0] -> reactions[0][1]
    reactions[1][0] -> reactions[1][1]
                ...

    :param reactions: list of lists of two np.array objects.
                The first np.array in each pair is the number
                of molecules in each reactant and the second
                is for the products
            ex. [[np.array([1,2,0]), np.array([0,0,1])]]
            represents A + 2B -> C
    :param k_values: a list of np.array objects with 2
                elements each. One for the reaction coefficient
                k for the forward reaction and one for the
                reverse
            ex. [[1,0], [2, 3]]
            represents k_1 = 1, k_-1 = 0
                       k_2 = 2, k_-2 = 3
    :return: function used by solve_ivp, a np.array representing
                all of the derivatives of the concentrations
                as functions of the concentrations

            ex. A + 2B <-> C
            output: d[A]/dt = -k1 [A][B]^2 + k-1 [C]
                    d[B]/dt = -0.5 k1 [A][B]^2 + 0.5 k-1 [C]
                    d[C]/dt = k1 [A][B]^2 - k-1 [C]
                where y = np.array([ [A], [B], [C] ])
                returning np.array([ d[A]/dt, d[B]/dt, d[C]/dt ])
    """
    equation_n = len(reactions)
    molecule_n = len(reactions[0][0])

    # assert equation_n == len(products)
    # assert equation_n == len(k_values)
    # assert len(k_values[0]) == 2
    # assert molecules == len(products[0])

    def f(t, y):
        y_diff = np.zeros(molecule_n)
        for molec in range(molecule_n):
            for eqn in range(equation_n):
                for k1 in range(2):  # looking at reactants (0) vs products (1)
                    if reactions[eqn][k1][molec] >= 1:
                        for k2 in range(2):  # looking at outward reaction (0) vs inward reaction (1)
                            y_diff[molec] -= (-1) ** k2 * k_values[eqn][k1 ^ k2] / reactions[eqn][k1][molec] * \
                                             np.prod([y[i] ** reactions[eqn][(k1 + k2) % 2][i]
                                                      if reactions[eqn][(k1 + k2) % 2][i] > 0 else 1 for i in
                                                      range(molecule_n)])
        return y_diff

    return f


def calc_qc(concentration, reactions):
    """
    Calculates the value Qc for a gives system of reactions
    for a given np.array of concentrations. When applied on
    the concentrations for a reaction reaching equilibrium
    this should equal Kc.

    Note: Function uses numpy so concentration can be a
        2d array to calculate Qc for various points in time

         [A]^a [B]^b ...
    Qc = ---------------
         [C]^c [D]^d ...

    For reaction aA + bB ... -> cC + dD ...

    :param concentration: np.array of concentrations
    :param reactions: list of lists of two np.array objects.
                The first np.array in each pair is the number
                of molecules in each reactant and the second
                is for the products
            ex. [[np.array([1,2,0]), np.array([0,0,1])]]
            represents A + 2B -> C
    :return: double Qc
    """
    overall_reactants = sum([x[0] for x in reactions])
    overall_products = sum([x[1] for x in reactions])
    return np.dot(concentration, overall_products) / np.dot(concentration, overall_reactants)


if __name__ == '__main__':
    reactions = [[np.array([1, 2, 0, 0]), np.array([0, 0, 1, 0])],
                 [np.array([0, 1, 1, 0]), np.array([0, 0, 0, 3])]]
    k_values = [[2, 1],
                [4, 0.1]]
    chem_initial = [1, 1, 0, 2]

    x_interval = [0, 5]
    T = np.arange(*x_interval, 0.01)

    t1 = time.time()
    sol = solve_ivp(eqn_to_odes(reactions, k_values), x_interval, chem_initial, t_eval=T)
    print(time.time() - t1)

    Qc = calc_qc(np.array([[sol.y[i][j] for i in range(len(chem_initial))] for j in range(len(T))]), reactions)

    plt.figure(1)
    for i in range(4):
        plt.plot(sol.t, sol.y[i], label=i)

    plt.legend()
    plt.grid()
    plt.title('Concentrations of System of Reaction')
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.xlim(x_interval)

    plt.figure(2)
    plt.title('$Q_c$')
    plt.ylabel('$Q_c$')
    plt.xlabel('t')
    plt.grid()
    plt.plot(T, Qc)
    plt.xlim(x_interval)
    plt.show()
