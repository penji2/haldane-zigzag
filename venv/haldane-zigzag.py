import random
import numpy as np
import matplotlib.pyplot as plt
import cmath

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


class Haldane:
    def __init__(self):
        self.t = 1
        self.lamb = 0.5
        self.V = 0
        self.n = 8  # dim of matrix


    """hamiltonian for a given k"""

    def hamiltonian(self, k):
        hamiltonian = np.zeros((self.n, self.n), dtype=complex)
        anan_odd = -self.V + 1j * self.lamb * (-np.exp(1j * k) + np.exp(-1j * k))
        anan_even = self.V + 1j * self.lamb * (-np.exp(-1j * k) + np.exp(1j * k))

        a1a2 = self.t * (1 + np.exp(-1j * k))
        a1a3 = 1j * self.lamb * (1 - np.exp(-1j * k))

        a2a3 = self.t
        a2a4 = 1j * self.lamb * (1 - np.exp(1j * k))

        a3a4 = self.t * (1 + np.exp(1j * k))
        a3a5 = 1j * self.lamb * (-1 + np.exp(1j * k))

        a4a5 = self.t
        a4a6 = 1j * self.lamb * (-1 + np.exp(-1j * k))
        for i in range(self.n):

            if i % 4 == 0:
                if i > 3:
                    hamiltonian[i][i - 2] = np.conj(hamiltonian[i - 2][i])
                    hamiltonian[i][i - 1] = np.conj(hamiltonian[i - 1][i])
                hamiltonian[i][i] = anan_odd
                if i + 1 < self.n:
                    hamiltonian[i][i + 1] = a1a2
                if i + 2 < self.n:
                    hamiltonian[i][i + 2] = a1a3

            elif i % 4 == 1:
                if i > 4:
                    hamiltonian[i][i - 2] = np.conj(hamiltonian[i - 2][i])
                hamiltonian[i][i - 1] = np.conj(hamiltonian[i - 1][i])
                hamiltonian[i][i] = anan_even
                if i + 1 < self.n:
                    hamiltonian[i][i + 1] = a2a3
                if i + 2 < self.n:
                    hamiltonian[i][i + 2] = a2a4

            elif i % 4 == 2:
                hamiltonian[i][i - 2] = np.conj(hamiltonian[i - 2][i])
                hamiltonian[i][i - 1] = np.conj(hamiltonian[i - 1][i])
                hamiltonian[i][i] = anan_odd
                if i + 1 < self.n:
                    hamiltonian[i][i + 1] = a3a4
                if i + 2 < self.n:
                    hamiltonian[i][i + 2] = a3a5

            elif i % 4 == 3:
                hamiltonian[i][i - 2] = np.conj(hamiltonian[i - 2][i])
                hamiltonian[i][i - 1] = np.conj(hamiltonian[i - 1][i])
                hamiltonian[i][i] = anan_even
                if i + 1 < self.n:
                    hamiltonian[i][i + 1] = a4a5
                if i + 2 < self.n:
                    hamiltonian[i][i + 2] = a4a6

        return hamiltonian

    def get_eigenvalues_for_plot(self):

        k_array = list()
        k = -np.pi
        k_step = 0.05
        i = 0
        whole_eig_energy = np.zeros(((int(2 * np.pi / k_step)) + 1, self.n), dtype=complex)
        while k < np.pi:
            whole_eig_energy[i] = (list(np.linalg.eigh(self.hamiltonian(k))[0]))

            k_array.append(k)
            k = k + k_step
            i += 1

        return k_array, whole_eig_energy

    def plot_data(self, xaxis, data_sets):

        label_font_size = 20



        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        for i in range(self.n):
            ax1.scatter(xaxis, data_sets[:, i], color='red')

        plt.title("Energy bands", fontsize=label_font_size)
        plt.xlabel("k", fontsize=label_font_size)
        plt.ylabel("E", fontsize=label_font_size)

        plt.show()


if __name__ == '__main__':
    test = Haldane()
    tmp = test.get_eigenvalues_for_plot()
    test.plot_data(tmp[0], tmp[1])
