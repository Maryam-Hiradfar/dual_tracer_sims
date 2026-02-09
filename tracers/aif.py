# tracers/aif.py
import numpy as np


RACLOPRIDE_DEFAULT = {
    "A1": 3.4,
    "A2": 1.4,
    "A3": 0.15,
    "lam1": 3.0,
    "lam2": 0.2,
    "lam3": 0.02
}

def raclopride_aif(t, A1 = 3.4, A2 = 1.4, A3 = 0.15, lam1 = 3.0, lam2 = 0.2, lam3 = 0.02, scale = 1.0):
    Cp = np.zeros_like(t)
    term1 = A1 * np.exp(-lam1 * t)
    term2 = A2 * np.exp(-lam2 * t)
    term3 = A3 * np.exp(-lam3 * t)
    Cp = scale * (term1 + term2 + term3)
  
    return Cp

FENG_FDG_DEFAULT = {
    "A1": 1854.66,
    "A2": 8.168,
    "A3": 2.731,
    "lam1": 20.031,
    "lam2": 0.355,
    "lam3": 0.0178,
    "tau": 0.27
}
def feng_aif(t_since_inj, A1 = 1854.66, A2 = 8.168, A3 = 2.731, lam1 = 20.031, lam2 = 0.355, lam3 = 0.0178, tau = 0.27, scale = 1.0):
    t_shift = np.maximum(t_since_inj - tau, 0.0)
    term1 = (A1 * t_shift - A2 - A3) * np.exp(-lam1 * t_shift)
    term2 = A2 * np.exp(-lam2 * t_shift)
    term3 = A3 * np.exp(-lam3 * t_shift)
    return scale * (term1 + term2 + term3)

PBR28_DEFAULT = {
    "A1": 22.0,
    "A2": 3.2,
    "A3": 0.25,
    "lam1": 3.1,
    "lam2": 0.23,
    "lam3": 0.015
}

def pbr28_aif(t, A1 = 22.0, A2 = 3.2, A3 = 0.25, lam1 = 3.1, lam2 = 0.23, lam3 = 0.015, scale = 1.0):
    #A1, A2, A3 units are kBq/mL-scale amplitudes
    #lam1, lam2, lam3 units are # 1/min decay rates

    return scale * (A1*  np.exp(-lam1 * t) +
            A2 *  np.exp(-lam2 * t) +
            A3 *  np.exp(-lam3 * t))
def estimated_injected_dose(aif_func, blood_volume_ml = 5000): 
    C_0  = aif_func(0.0)  # AIF value at time zero (peak concentration)
    dose_kbq = C_0 * blood_volume_ml  # Total activity in kBq
    dose_mCi = dose_kbq / 37000.0  # Convert kBq to mCi (1 mCi = 37,000 kBq)
    return dose_kbq, dose_mCi
