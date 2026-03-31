# kinetics/twotcm.py
import numpy as np

def simulate_2tcm(K1, k2, k3, k4, Cp, t):
    dt = t[1] - t[0]
    n = len(t)

    C1 = np.zeros(n)
    C2 = np.zeros(n)

    for i in range(1, n):
        dC1 = K1 * Cp[i-1] - (k2 + k3) * C1[i-1] + k4 * C2[i-1]
        dC2 = k3 * C1[i-1] - k4 * C2[i-1]

        C1[i] = C1[i-1] + dC1 * dt
        C2[i] = C2[i-1] + dC2 * dt

    return C1 + C2, C1, C2
