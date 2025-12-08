# tracers/aif.py
import numpy as np

def raclopride_aif(t):
    A = [3.4, 1.4, 0.15]
    lam = [3.0, 0.2, 0.02]
    Cp = np.zeros_like(t)
    for Ai, li in zip(A, lam):
        Cp += Ai * np.exp(-li * t)
    return Cp

def feng_aif(t_since_inj, A1, A2, A3, lam1, lam2, lam3, tau, scale = 1.0):
    t_shift = np.maximum(t_since_inj - tau, 0.0)
    term1 = (A1 * t_shift - A2 - A3) * np.exp(-lam1 * t_shift)
    term2 = A2 * np.exp(-lam2 * t_shift)
    term3 = A3 * np.exp(-lam3 * t_shift)
    return scale * (term1 + term2 + term3)
