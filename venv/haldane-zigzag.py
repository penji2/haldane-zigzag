import random
import numpy as np
import matplotlib.pyplot as plt
import cmath
from mpl_toolkits.mplot3d import Axes3D


class Haldane:
    def __init__(self):
        self.t = 1
        self.lamb = 0.4
        self.V = 0.2
        self.n = 7  # dim of matrix


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


        print(hamiltonian)
        return hamiltonian

    def get_eigenvalues_for_plot3d(self):
        eigenvalues_array_0, eigenvalues_array_1, kx_array, ky_array = list(), list(), list(), list()
        k_step = 0.05
        for kx in np.arange(-np.pi, np.pi, k_step):  # defining grid loop
            for ky in np.arange(-np.pi, np.pi, k_step):
                k1 = np.dot(np.array([kx, ky]), self.a1)
                k2 = np.dot(np.array([kx, ky]), self.a2)
                eigenvalues_array_0.append(np.linalg.eigh(self.define_hamiltonian(k1, k2))[0][0])
                eigenvalues_array_1.append(np.linalg.eigh(self.define_hamiltonian(k1, k2))[0][1])
                kx_array.append(kx)
                ky_array.append(ky)

        return kx_array, ky_array, eigenvalues_array_0, eigenvalues_array_1


if __name__ == '__main__':
    test = Haldane()
    test.hamiltonian(np.pi / 2)