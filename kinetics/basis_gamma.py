# kinetics/basis_gamma.py
import numpy as np

def gamma_shape(t, t0, tau, scale= 1.0):
    x = np.maximum(t - t0, 0)
    return scale * (x / tau) * np.exp(-x / tau)

def build_gamma_library(t, n_t0, n_tau, scale=1.0):
    t0_values  = np.linspace(0, float(t[-1]) - 10, n_t0)
    tau_values = np.linspace(0.3, 40, n_tau)

    lib = []
    for t0 in t0_values:
        for tau in tau_values:
            lib.append(gamma_shape(t, t0, tau, scale=scale))

    lib = np.array(lib)
    return lib, lib.T     # gamma_lib, Gamma
def build_gamma_library_for_two_tracers(t, n_t0_1, n_tau_1, n_t0_2, n_tau_2, scale_1= 2, scale_2=3):
    lib_1, Gamma_1 = build_gamma_library(t, n_t0_1, n_tau_1, scale=scale_1)
    lib_2, Gamma_2 = build_gamma_library(t, n_t0_2, n_tau_2, scale=scale_2)
    return lib_1, Gamma_1, lib_2, Gamma_2
# def build_gamma_library_for_two_auto_scale(t, n_t0_1, n_tau_1, n_t0_2, n_tau_2, scale_1=1.0, scale_2=1.0):
# """
# uses the peak finder to set the average peak of the gamma functions in each library to the provided scale for that tracer, then builds libraries by 
# varying 
# """
def build_gamma_library_for_two_specific_injection_time_points(t, t0_values_1, n_tau_1, t0_values_2, n_tau_2, scale_1= 1, scale_2=1):
    lib_1 = []
    for t0 in t0_values_1:
        tau_values = np.linspace(0.3, 40, n_tau_1)
        for tau in tau_values:
            lib_1.append(gamma_shape(t, t0, tau, scale=scale_1))
    lib_1 = np.array(lib_1)
    Gamma_1 = lib_1.T

    lib_2 = []
    for t0 in t0_values_2:
        tau_values = np.linspace(0.3, 40, n_tau_2)
        for tau in tau_values:
            lib_2.append(gamma_shape(t, t0, tau, scale=scale_2))
    lib_2 = np.array(lib_2)
    Gamma_2 = lib_2.T

    return lib_1, Gamma_1, lib_2, Gamma_2