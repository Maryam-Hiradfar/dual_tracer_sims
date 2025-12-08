# core/noise.py
#####Units#######
# y_clean: kBq/mL
# frame_durs: minutes
# V_eff: mL
# S: cps per kBq/mL

import numpy as np

def uexplorer_poisson(y_clean, frame_durs, V_eff=20, S=174.0,  rng=None):
    """
    Poisson noise model for PET:
    y_clean: true physical TAC values per frame
    frame_durs: frame durations in minutes
    """
    if rng is None:
        rng = np.random.default_rng()

    dur_sec = frame_durs * 60.0
    counts = S * V_eff * y_clean * dur_sec
    counts[counts < 0] = 0

    noisy_counts = rng.poisson(counts)
    return noisy_counts / (S * V_eff * dur_sec) #the division is to convert back to kBq/mL
