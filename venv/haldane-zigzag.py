import random
import numpy as np
import matplotlib.pyplot as plt
import cmath
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


class Haldane:
    def __init__(self):
        self.t = 1
        self.lamb = 0.1
        self.V = 0
        self.n = 6  # dim of matrix


    """hamiltonian for a given k"""

    def hamiltonian(self, k):
        hamiltonian = np.zeros(shape=(self.n, self.n), dtype=complex)

        a1a2 = self.t * (1 + np.exp(-1j * k))
        a1a3 = 1j * self.lamb * (1 - np.exp(-1j * k))

        a2a3 = self.t
        a2a4 = 1j * self.lamb * (1 - np.exp(1j * k))

        a3a4 = self.t * (1 + np.exp(1j * k))
        a3a5 = 1j * self.lamb * (-1 + np.exp(1j * k))

        a4a5 = self.t
        a4a6 = 1j * self.lamb * (-1 + np.exp(-1j * k))
        for i in range(self.n):
            if i % 2 == 0:
                hamiltonian[i][i] = -self.V + 2 * self.lamb * np.sin(k)
            if i % 2 == 1:
                hamiltonian[i][i] = self.V - 2 * self.lamb * np.sin(k)

            if i % 4 == 0:
                if i > 3:
                    hamiltonian[i][i - 2] = np.conj(hamiltonian[i - 2][i])
                    hamiltonian[i][i - 1] = np.conj(hamiltonian[i - 1][i])

                if i + 1 < self.n:
                    hamiltonian[i][i + 1] = a1a2
                if i + 2 < self.n:
                    hamiltonian[i][i + 2] = a1a3

            if i % 4 == 1:
                if i > 4:
                    hamiltonian[i][i - 2] = np.conj(hamiltonian[i - 2][i])
                    hamiltonian[i][i - 1] = np.conj(hamiltonian[i - 1][i])

                if i + 1 < self.n:
                    hamiltonian[i][i + 1] = a2a3
                if i + 2 < self.n:
                    hamiltonian[i][i + 2] = a2a4

            if i % 4 == 2:
                hamiltonian[i][i - 2] = np.conj(hamiltonian[i - 2][i])
                hamiltonian[i][i - 1] = np.conj(hamiltonian[i - 1][i])

                if i + 1 < self.n:
                    hamiltonian[i][i + 1] = a3a4
                if i + 2 < self.n:
                    hamiltonian[i][i + 2] = a3a5

            if i % 4 == 3:
                hamiltonian[i][i - 2] = np.conj(hamiltonian[i - 2][i])
                hamiltonian[i][i - 1] = np.conj(hamiltonian[i - 1][i])

                if i + 1 < self.n:
                    hamiltonian[i][i + 1] = a4a5
                if i + 2 < self.n:
                    hamiltonian[i][i + 2] = a4a6
        return hamiltonian

    def get_eigenvalues_for_plot(self):
        k_array, whole_eig_energy = list(), list()
        k = 0
        k_step = 0.008
        while k < 2 * np.pi:
            whole_eig_energy.append(np.linalg.eigvalsh(self.hamiltonian(k)))

            k_array.append(k)
            k = k + k_step

        return k_array, whole_eig_energy

    def plot_data(self, xaxis, data_sets):
        label_font_size = 20
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        for i in range(self.n):
            ax1.plot(xaxis, np.transpose(data_sets)[i], 'k-', linewidth=0.6, markersize=0.4)

        plt.title("Energy bands, nontrivial, 40 atoms", fontsize=label_font_size)
        plt.xlabel("k", fontsize=label_font_size)
        plt.ylabel("E", fontsize=label_font_size)
        plt.show()


if __name__ == '__main__':
    test = Haldane()
    tmp = test.get_eigenvalues_for_plot()
    test.plot_data(tmp[0], tmp[1])
