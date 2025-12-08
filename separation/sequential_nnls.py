# separation/sequential_nnls.py
import numpy as np
from scipy.optimize import nnls

def sequential_unmix(y_meas, Phi_rac, Phi_fdg, t_frames, t_cut):
    early_mask = t_frames <= t_cut

    w_rac, _ = nnls(Phi_rac[early_mask, :], y_meas[early_mask])
    rac_est = Phi_rac @ w_rac

    resid = np.clip(y_meas - rac_est, 0, None)
    w_fdg, _ = nnls(Phi_fdg, resid)
    fdg_est = Phi_fdg @ w_fdg

    return rac_est, fdg_est
