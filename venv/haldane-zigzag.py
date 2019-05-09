import random
import numpy as np
import matplotlib.pyplot as plt
import cmath
from mpl_toolkits.mplot3d import Axes3D


class Haldane:
    def __init__(self):
        self.t = 1
        self.lamb = 0.1
        self.V = 0

        self.a1 = 0.5 * np.array([-3 ** 0.5, 3])  # translation vector1 in position domain
        self.a2 = 0.5 * np.array([3 ** 0.5, 3])  # translation vector2 in position domain

    """hamiltonian for a given k"""

    def define_hamiltonian(self, k1, k2):
        ab = self.t * (1 + np.exp(-1j * k1) + np.exp(-1j * k2))
        ba = np.conj(ab)
        aa = self.V + 1j * self.lamb * (-np.exp(1j * (k1 - k2)) + np.exp(1j * k1) - np.exp(1j * k2)
                                        + np.exp(-1j * (k1 - k2)) - np.exp(-1j * k1) + np.exp(-1j * k2))
        # bb = -self.V + self.lamb * (np.exp(1j * (k1 - k2)) + np.exp(1j * k1) - np.exp(1j * k2)
        #                             - np.exp(1j * (k1 - k2)) + np.exp(-1j * k1) - np.exp(-1j * k2))
        bb = -aa

        hamiltonian = np.array([[aa, ab],
                                [ba, bb]])
        return hamiltonian

    @staticmethod
    def finite_hamiltonian(n, t, dt):
        hamiltonian = np.zeros((n, n))
        for i in range(n - 1):
            if i % 2 == 0:
                hamiltonian[i][i + 1] = t + dt
                hamiltonian[i + 1][i] = t + dt
            else:
                hamiltonian[i][i + 1] = t - dt
                hamiltonian[i + 1][i] = t - dt
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
    test.finite_hamiltonian(5, 1, 0.2)
