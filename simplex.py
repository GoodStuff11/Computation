import numpy as np


class Simplex:
    def __init__(self, c, k, A, b):
        """
        min c^T x + k
        st  Ax = b
            x >= 0

        :param c: np.array
        :param A: list of np.array
        :param b: np.array
        """
        self.display = False

        self.eqn = len(b)
        self.vars = len(c)

        # augmented matrix
        self.A = A
        self.Ab = [np.concatenate((A[i], np.array(b[i])), axis=None) for i in range(len(A))]
        self.c = c
        self.b = b
        self.k = k

        self.basis = []

    def show_steps(self, boolean):
        assert type(boolean) == bool, "Input must be boolean: " + str(boolean)
        self.display = boolean

    def solve_initial_basis(self):
        """
        Uses simplex on auxiliary problem to find a feasible basis
         for the LP. Changes self.basis

         Raises an error if the primal program is infeasible
        """
        # new_basis = [i + 1 for i, val in enumerate(self.c) if val > 0]
        # other = [i + 1 for i, val in enumerate(self.c) if val <= 0]
        # if len(new_basis) >= self.eqn:
        #     self.basis = new_basis[:self.eqn]
        # else:
        #     self.basis = sorted(new_basis + other[:self.eqn - len(new_basis)])
        c = np.array([0 if i < self.vars else -1 for i in range(self.vars + self.eqn)])
        A = np.concatenate(([self.A[row] if self.b[row] > 0
                             else -self.A[row] for row in range(self.eqn)],
                            np.identity(self.eqn)), axis=1)
        b = np.abs(self.b)

        p = Simplex(c, 0, A, b)
        p.show_steps(self.display)

        # detect for negative b value
        p.set_basis([x for x in range(self.vars + 1, self.vars + self.eqn + 1)])
        p.solve()
        print('-' * 100)
        x = p.get_BFS()

        assert p.k == 0, "Infeasible Primal Program"

        self.basis = p.get_basis()

    def detect_basis(self):
        """
        Given an LP, sets self.basis to be what it should be so
        that update_canonical_form() will do nothing. If this is
        not possible, prints a message
        :return:
        """
        raise NotImplementedError

    def get_basis(self):
        return self.basis

    def get_augmented_matrix(self):
        """
        Returns augmented matrix (A|b) from the system of equations
        Ax = b
        :return: np.array
        """
        return np.array(self.Ab)

    def get_objective_function(self):
        """
        Returns the c and k values from the objective function
         cx + b
        :return: (np.array, float)
        """
        return self.c, self.k

    def step(self):
        if self.find_new_basis():
            self.update_canonical_form()

    def solve(self):
        """
        Solves LP, must have a feasible basis initially
        :return:
        """
        assert len(self.basis) == self.eqn, "Invalid basis:" + str(self.basis)

        if self.display:
            print("Updating to basis:", self.basis)
            print(self)
            print('-' * 100)

        self.update_canonical_form()
        if self.display:
            print("New basis:", self.basis)
            print(self)
            print('-' * 100)

        self.update_canonical_form()
        while self.find_new_basis():
            self.update_canonical_form()
            if self.display:
                print("New basis:", self.basis)
                print(self)
                print('-' * 100)

    def set_basis(self, basis):
        """
        sets a new basis, making sure that it is valid
        :param basis: a numpy array
        """
        basis = list(set(basis))
        if len(basis) <= self.eqn:
            self.basis = sorted(basis)
        else:
            print('invalid basis')

    def find_new_basis(self):
        """
        Use Blands rule to find new basis.
        Changes self.basis, does NOT change self.Ab or self.c
        :return: False if unbounded
                 True if found valid next basis
        """
        # minimum value greater than 0
        if all([ci <= 0.0001 for ci in self.c]):
            return False

        # print(self.c)
        enter = np.where(self.c > 0)[0][0] + 1
        # min b/Aij for Aij > 0 and b > 0
        select_thing = np.array([self.Ab[i][enter - 1] / self.Ab[i][-1]
                                 if self.Ab[i][enter - 1] > 0 and self.Ab[i][-1] > 0
                                 else -1 for i in range(self.eqn)])
        if all(select_thing == -1):
            print("UNBOUNDED")
            return
        else:
            exit = self.basis[int(np.argmax(select_thing))]

        # print(self.basis, enter, exit)
        self.basis.remove(exit)
        self.basis.append(enter)
        self.basis.sort()

        # print("New basis:", self.basis)
        return True

    def update_canonical_form(self):
        """
        Adjusts self.Ab and self.c so that it has the basis self.basis
        """
        # print(np.array(self.Ab))
        for i, base in enumerate(self.basis):
            base -= 1
            # making sure that the basis isn't already self.basis
            if not all(self.Ab[k][base] == 0 if k != i else self.Ab[k][base] == 1 for k in range(self.eqn)):
                if self.Ab[i][base] == 0:
                    # find row where it is not 0 then swap
                    for row in range(self.eqn):
                        if self.Ab[row][base] != 0:
                            # swap equations
                            temp = self.Ab[i]
                            self.Ab[i] = self.Ab[row]
                            self.Ab[row] = temp
                            break

                # zero division error means that it is infeasible
                self.Ab[i] = self.Ab[i] / self.Ab[i][base]
                # print(np.array(self.Ab))
                for row in range(self.eqn):
                    if row != i:
                        self.Ab[row] = self.Ab[row] - self.Ab[row][base] * self.Ab[i]
                        # print(np.array(self.Ab))
                    if self.display:
                        print("step", row)
                        print(np.array(self.Ab))
                        print()

        c = self.c
        for i, base in enumerate(self.basis):
            base -= 1
            self.c = self.c - c[base] * self.Ab[i][:-1]
            self.k += c[base] * self.Ab[i][-1]

    def get_BFS(self):
        """
        Given a known basis self.basis and self.Ab and self.c adjusted
        to it, returns the corresponding BFS or None if the basis is invalid
        or if the solution is infeasible
        :return: None if basis is invalid

        """
        if len(self.basis) == len(self.Ab):
            x = [0 for i in range(self.vars)]
            for i, base in enumerate(self.basis):
                x[base - 1] = self.Ab[i][-1]

            assert all(np.array(x) >= 0), "Infeasible Solution:" + str(np.array(x))
            return x
        print('Invalid basis: ' + str(self.basis))

    def __str__(self):
        return str(self.c) + 'x + ' + str(self.k) + '\n' + str(np.array(self.Ab))


if __name__ == '__main__':
    A = [np.array([1, 2, -1, 0]),
        np.array([0, 5, -2, -1])]

    b = np.array([4, 1])
    c = np.array([1, 2, 3, 4])

    # A = [np.array([2, 5, 5, 1]),
    #      np.array([-1, 1, 0, 2]),
    #      np.array([4, -2, -1, 3])]
    #
    # b = np.array([3, 2, -2])
    # c = np.array([4, -2, 1, 0])

    p1 = Simplex(c, 0, A, b)
    p1.show_steps(True)
    p1.solve_initial_basis()
    p1.solve()
