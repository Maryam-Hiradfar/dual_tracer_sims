# core/noise.py
#####Units#######
# y_clean: kBq/mL
# frame_durs: minutes
# V_eff: mL
# S: cps per kBq/mL

import numpy as np

#----------------------------
# Scanner presets
#----------------------------
SCANNERS = {
    "panorama_gs" :{
        "S_kcps_per_MBq": 176.3, 
        "description": "United Imaging's Panorama PET/CT scanner, with a sensitivity of 176.3 kcps/MBq (or 176.3 cps/kBq)."
    }
}

#----------------------------
# Volumes
#----------------------------
VOLUMES = {
    # ROI volumes (from anatomical atlases)
    "striatum":       10.0,
    "putamen":         6.0,
    "caudate":         4.0,
    "cerebellum":     30.0,
    "frontal_cortex": 25.0,
    "whole_brain":  1400.0,

    # Typical PET voxel volumes
    "voxel_2mm": 0.008,    # 2×2×2 mm
    "voxel_3mm": 0.027,    # 3×3×3 mm
    "voxel_4mm": 0.064,    # 4×4×4 mm
}


def _activity_to_counts(y_clean_kBq_mL, V_mL, S_kcps_per_MBq, dur_sec):
    """
    Convert activity concentration to expected photon counts.

    y_clean (kBq/mL) → activity in voxel/ROI (MBq) → count rate (cps) → counts

    Parameters
    ----------
    y_clean_kBq_mL : np.ndarray
        Activity concentration per frame (kBq/mL).
    V_mL : float
        Volume in mL (voxel or ROI).
    S_kcps_per_MBq : float
        Scanner sensitivity in kcps/MBq.
    dur_sec : np.ndarray
        Frame durations in seconds.

    Returns
    -------
    expected_counts : np.ndarray
        Expected number of detected counts per frame.
    """
    A_MBq = y_clean_kBq_mL * V_mL / 1000.0        # kBq/mL × mL → kBq → MBq
    count_rate_cps = S_kcps_per_MBq * 1000.0 * A_MBq  # kcps/MBq → cps/MBq × MBq → cps
    expected_counts = count_rate_cps * dur_sec
    return np.clip(expected_counts, 0, None)


def _counts_to_concentration(counts, V_mL, S_kcps_per_MBq, dur_sec):
    """
    Convert detected counts back to activity concentration (kBq/mL).
    Inverse of _activity_to_counts.
    """
    return counts / (S_kcps_per_MBq * 1000.0 * (V_mL / 1000.0) * dur_sec)


def add_poisson_noise(y_clean, frame_durs, V_mL, S_kcps_per_MBq, rng=None):
    """
    Apply Poisson noise to a clean TAC.

    Parameters
    ----------
    y_clean : np.ndarray, shape (T,)
        True activity concentration per frame (kBq/mL).
    frame_durs : np.ndarray, shape (T,)
        Frame durations in minutes.
    V_mL : float
        Volume in mL. Use ROI volume for ROI-level,
        voxel volume for voxel-level simulation.
    S_kcps_per_MBq : float
        Scanner sensitivity (kcps/MBq).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    y_noisy : np.ndarray, shape (T,)
        Noisy activity concentration (kBq/mL).
    expected_counts : np.ndarray, shape (T,)
        Expected counts per frame (useful for diagnostics).
    """
    if rng is None:
        rng = np.random.default_rng()

    dur_sec = frame_durs * 60.0
    expected_counts = _activity_to_counts(y_clean, V_mL, S_kcps_per_MBq, dur_sec)
    noisy_counts = rng.poisson(expected_counts)
    y_noisy = _counts_to_concentration(noisy_counts, V_mL, S_kcps_per_MBq, dur_sec)

    return y_noisy, expected_counts


# ──────────────────────────────────────────────
# Convenience functions
# ──────────────────────────────────────────────

def add_noise_roi(y_clean, frame_durs, roi_name, scanner_name="panorama_gs", rng=None):
    """
    Add Poisson noise at ROI level.

    Parameters
    ----------
    y_clean : np.ndarray
        Clean TAC in kBq/mL.
    frame_durs : np.ndarray
        Frame durations in minutes.
    roi_name : str
        Key from VOLUMES dict (e.g., "striatum", "putamen").
    scanner_name : str
        Key from SCANNERS dict.
    rng : np.random.Generator, optional

    Returns
    -------
    y_noisy : np.ndarray
    expected_counts : np.ndarray

    Example
    -------
    >>> y_noisy, counts = add_noise_roi(y_clean, frame_durs, "striatum", "panorama_gs")
    """
    V_mL = VOLUMES[roi_name]
    S = SCANNERS[scanner_name]["S_kcps_per_MBq"]
    return add_poisson_noise(y_clean, frame_durs, V_mL, S, rng)


def add_noise_voxel(y_clean, frame_durs, voxel_size_mm=4.0,
                    scanner_name="panorama_gs", rng=None):
    """
    Add Poisson noise at voxel level.

    Parameters
    ----------
    y_clean : np.ndarray
        Clean TAC in kBq/mL.
    frame_durs : np.ndarray
        Frame durations in minutes.
    voxel_size_mm : float
        Isotropic voxel edge length in mm. Default 4mm.
    scanner_name : str
        Key from SCANNERS dict.
    rng : np.random.Generator, optional

    Returns
    -------
    y_noisy : np.ndarray
    expected_counts : np.ndarray

    Example
    -------
    >>> y_noisy, counts = add_noise_voxel(y_clean, frame_durs, voxel_size_mm=3.0)
    """
    V_mL = (voxel_size_mm ** 3) / 1000.0  # mm³ → mL
    S = SCANNERS[scanner_name]["S_kcps_per_MBq"]
    return add_poisson_noise(y_clean, frame_durs, V_mL, S, rng)


# ──────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────

def noise_summary(y_clean, frame_durs, V_mL, S_kcps_per_MBq):
    """
    Print a summary of expected counts and noise levels per frame.
    Useful for sanity checking your simulation setup.
    """
    dur_sec = frame_durs * 60.0
    expected_counts = _activity_to_counts(y_clean, V_mL, S_kcps_per_MBq, dur_sec)
    cv = np.where(expected_counts > 0, 1.0 / np.sqrt(expected_counts), np.inf)

    print(f"{'Frame':>5} | {'Duration(s)':>11} | {'y_clean(kBq/mL)':>15} | "
          f"{'Exp. Counts':>12} | {'CV (%)':>8}")
    print("-" * 72)
    for i in range(len(y_clean)):
        print(f"{i:5d} | {dur_sec[i]:11.1f} | {y_clean[i]:15.3f} | "
              f"{expected_counts[i]:12.0f} | {cv[i]*100:8.2f}")

    print("-" * 72)
    print(f"Total expected counts: {expected_counts.sum():,.0f}")
    print(f"Mean CV: {cv.mean()*100:.2f}%")