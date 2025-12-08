# kinetics/basis_gamma.py
import numpy as np

def gamma_shape(t, t0, tau, scale=1.0):
    x = np.maximum(t - t0, 0)
    return scale * (x / tau) * np.exp(-x / tau)

def build_gamma_library(t, n_t0=30, n_tau=30):
    t0_values  = np.linspace(0, float(t[-1]) - 10, n_t0)
    tau_values = np.linspace(0.3, 40, n_tau)

    lib = []
    for t0 in t0_values:
        for tau in tau_values:
            lib.append(gamma_shape(t, t0, tau))

    lib = np.array(lib)
    return lib, lib.T     # gamma_lib, Gamma
