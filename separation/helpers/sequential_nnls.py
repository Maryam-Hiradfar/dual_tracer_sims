# separation/sequential_nnls.py
import numpy as np
from scipy.optimize import nnls

# def sequential_unmix(y_meas, Phi_rac, Phi_fdg, t_frames, t_cut):
#     early_mask = t_frames <= t_cut

#     w_rac, _ = nnls(Phi_rac[early_mask, :], y_meas[early_mask])
#     rac_est = Phi_rac @ w_rac

#     resid = np.clip(y_meas - rac_est, 0, None)
#     w_fdg, _ = nnls(Phi_fdg, resid)
#     fdg_est = Phi_fdg @ w_fdg

    # return rac_est, fdg_est
def nnls_l2 (Phi, y, alpha):
    """
    NNLS with L2 Regularization (Tikohonov)
    Solves min_w ||Phi w - y||^2 + alpha * ||w||^2  subject to w >= 0
    Parameters
    _________
    Phi : np.ndarray, shape (T, n)
        Basis matrix.
    y : np.ndarray, shape (T,)
        Measured signal.
    alpha : float
        Regularization strength.
    Returns
    _______
    w : np.ndarray, shape (n,)
        Non-negative coefficients.
    Original system (T=6 time points, n=4 basis functions):

    ┌                 ┐       ┌      ┐
    │ Φ₁₁  Φ₁₂  Φ₁₃  Φ₁₄ │       │  y₁  │
    │ Φ₂₁  Φ₂₂  Φ₂₃  Φ₂₄ │       │  y₂  │
    │ Φ₃₁  Φ₃₂  Φ₃₃  Φ₃₄ │       │  y₃  │
    │ Φ₄₁  Φ₄₂  Φ₄₃  Φ₄₄ │  w ≈  │  y₄  │
    │ Φ₅₁  Φ₅₂  Φ₅₃  Φ₅₄ │       │  y₅  │
    │ Φ₆₁  Φ₆₂  Φ₆₃  Φ₆₄ │       │  y₆  │
    └                 ┘       └      ┘

    Augmented system (add 4 penalty rows):

    ┌                 ┐       ┌      ┐
    │ Φ₁₁  Φ₁₂  Φ₁₃  Φ₁₄ │       │  y₁  │  ← data fit
    │ Φ₂₁  Φ₂₂  Φ₂₃  Φ₂₄ │       │  y₂  │
    │ Φ₃₁  Φ₃₂  Φ₃₃  Φ₃₄ │       │  y₃  │
    │ Φ₄₁  Φ₄₂  Φ₄₃  Φ₄₄ │  w ≈  │  y₄  │
    │ Φ₅₁  Φ₅₂  Φ₅₃  Φ₅₄ │       │  y₅  │
    │ Φ₆₁  Φ₆₂  Φ₆₃  Φ₆₄ │       │  y₆  │
    │─────────────────│       │──────│
    │  α    0    0    0  │       │   0  │  ← penalty: want w₁ ≈ 0
    │  0    α    0    0  │       │   0  │  ← penalty: want w₂ ≈ 0
    │  0    0    α    0  │       │   0  │  ← penalty: want w₃ ≈ 0
    │  0    0    0    α  │       │   0  │  ← penalty: want w₄ ≈ 0
    └                 ┘       └      ┘

    The penalty rows say: "each weight times α should be close to 0"
    Larger α → stronger pull toward zero → smoother fit

    """
    n_basis  = Phi.shape[1]
    # Augment the system for L2 regularization
    Phi_aug = np.vstack([Phi, alpha * np.eye(n_basis)])
    # The target is augmented with zeros for the regularization part
    y_aug = np.concatenate([y, np.zeros(n_basis)])
    # Solve the augmented NNLS problem
    w, _ = nnls(Phi_aug, y_aug)
    return w
# def sequential_unmix_l2(y_meas, Phi_rac, Phi_fdg, t_frames, t_cut, alpha):
#     early_mask = t_frames <= t_cut

#     w_rac = nnls_l2(Phi_rac[early_mask, :], y_meas[early_mask], alpha)
#     rac_est = Phi_rac @ w_rac

#     resid = np.clip(y_meas - rac_est, 0, None)
#     w_fdg = nnls_l2(Phi_fdg, resid, alpha)
#     fdg_est = Phi_fdg @ w_fdg

#     return rac_est, fdg_est
# def joint_unmix(y_meas, Phi_rac, Phi_fdg, alpha):
#     Phi_joint = np.hstack([Phi_rac, Phi_fdg])
#     #w_joint, _ = nnls_l2(Phi_joint, y_meas, alpha)   # ← unpack with _ for regular nnls
#     w_joint = nnls_l2(Phi_joint, y_meas, alpha)   # ← unpack with _

#     n_rac = Phi_rac.shape[1]
#     rac_est = Phi_rac @ w_joint[:n_rac]
#     fdg_est = Phi_fdg @ w_joint[n_rac:]
#     return rac_est, fdg_est



# def joint_unmix(y_meas, Phi_rac, Phi_fdg, t_frames, t_cut,
#                      n_top=5, alpha_early=1, alpha_joint=1):
#     """
#     Maryam Method 1: Learn relevant RAC bases, then jointly fit.

#     Stage 1: Fit RAC to single-tracer region (before t_cut)
#              Select top-n most relevant basis functions
#     Stage 2: Joint NNLS on mixed region using
#              reduced RAC basis + full FDG basis

#     Parameters
#     ----------
#     y_meas : np.ndarray, shape (T,)
#         Combined noisy TAC.
#     Phi_rac : np.ndarray, shape (T, n_rac)
#         Full RAC basis matrix.
#     Phi_fdg : np.ndarray, shape (T, n_fdg)
#         Full FDG basis matrix.
#     t_frames : np.ndarray, shape (T,)
#         Time points (minutes).
#     t_cut : float
#         Injection time of second tracer.
#     n_top : int
#         Number of top RAC basis functions to keep.
#     alpha_early : float
#         Regularization for Stage 1 (early fit).
#     alpha_joint : float
#         Regularization for Stage 2 (joint fit).

#     Returns
#     -------
#     rac_est : np.ndarray, shape (T,)
#         Estimated RAC TAC.
#     fdg_est : np.ndarray, shape (T,)
#         Estimated FDG TAC.
#     info : dict
#         Diagnostics.
#     """
#     early_mask = t_frames <= t_cut
#     late_mask = t_frames > t_cut

#     # ──────────────────────────────────────────
#     # Stage 1: Fit RAC to early (single-tracer) region
#     # ──────────────────────────────────────────
#     w_early = nnls_l2(Phi_rac[early_mask, :], y_meas[early_mask], alpha_early)

#     # Find the top-n basis functions by weight magnitude
#     top_indices = np.argsort(w_early)[::-1][:n_top]
#     top_indices = np.sort(top_indices)  # keep in original order

#     # Reduced RAC basis (only the top-n columns)
#     Phi_rac_reduced = Phi_rac[:, top_indices]

#     # ──────────────────────────────────────────
#     # Stage 2: Joint NNLS on mixed region
#     # ──────────────────────────────────────────

#     # Build joint basis for late region only
#     Phi_joint_late = np.hstack([
#         Phi_rac_reduced[late_mask, :],   # reduced RAC bases
#         Phi_fdg[late_mask, :],           # full FDG bases
#     ])
#     y_late = y_meas[late_mask]

#     # Regularized joint NNLS
#     w_joint = nnls_l2(Phi_joint_late, y_late, alpha_joint)

#     # Split weights
#     w_rac_joint = w_joint[:n_top]
#     w_fdg_joint = w_joint[n_top:]

#     # ──────────────────────────────────────────
#     # Reconstruct full TACs (all time points)
#     # ──────────────────────────────────────────
#     rac_est = Phi_rac_reduced @ w_rac_joint
#     fdg_est = Phi_fdg @ w_fdg_joint

#     info = {
#         "top_indices": top_indices,
#         "w_early_full": w_early,
#         "w_early_top": w_early[top_indices],
#         "w_rac_joint": w_rac_joint,
#         "w_fdg_joint": w_fdg_joint,
#         "n_rac_original": Phi_rac.shape[1],
#         "n_rac_reduced": n_top,
#         "n_fdg": Phi_fdg.shape[1],
#     }

#     return rac_est, fdg_est, info




