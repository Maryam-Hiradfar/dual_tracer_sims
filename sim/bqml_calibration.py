"""Dose-referenced Bq/mL PET simulation from measured, scan-start decay-corrected IDIFs.

The empirical IDIF is whole blood (not parent plasma). Tissue mode uses the
repository's fixed 2TCM *as an approximation*, not quantitative kinetic inference.
Input data remain local; never add patient-derived CSVs to a public repository.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kinetics.models.twotcm import simulate_2tcm

MCI_TO_MBQ = 37.0
HALF_LIVES_MIN = {"FDG": 109.8, "PBR28": 20.4}
REFERENCE_DOSES_MCI = {"FDG": 4.91, "PBR28": 4.98}
APPROX_TISSUE_PARAMS = {
    "FDG": dict(K1=0.12, k2=0.25, k3=0.06, k4=0.0),
    "PBR28": dict(K1=0.18, k2=0.25, k3=0.04, k4=0.01),
}


@dataclass(frozen=True)
class MeasuredIDIF:
    tracer: str
    start_min: np.ndarray
    end_min: np.ndarray
    bqml: np.ndarray
    reference_dose_mci: float

    @classmethod
    def from_csv(cls, path: str | Path, tracer: str) -> "MeasuredIDIF":
        tracer = tracer.upper()
        if tracer not in REFERENCE_DOSES_MCI:
            raise ValueError(f"unsupported tracer: {tracer}")
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if not rows or any(not {"TimeStart", "TimeEnd", "aorta"}.issubset(row) for row in rows):
            raise ValueError("CSV must have TimeStart, TimeEnd (seconds), and aorta (Bq/mL)")
        start = np.array([float(row["TimeStart"]) / 60 for row in rows])
        end = np.array([float(row["TimeEnd"]) / 60 for row in rows])
        values = np.array([float(row["aorta"]) for row in rows])
        if (not np.isfinite(start).all() or not np.isfinite(end).all()
                or not np.isfinite(values).all() or np.any(values < 0)
                or np.any(end <= start) or np.any(np.diff(start) <= 0)
                or np.any(start[1:] < end[:-1] - 1e-8)
                or abs(start[0]) > 1e-8):
            raise ValueError("invalid, overlapping, negative, or nonzero-start IDIF frames")
        return cls(tracer, start, end, values, REFERENCE_DOSES_MCI[tracer])

    @property
    def mid_min(self) -> np.ndarray:
        return (self.start_min + self.end_min) / 2

    @property
    def coverage_min(self) -> float:
        return float(self.end_min[-1])

    def decay_corrected_bqml(self, relative_time_min: np.ndarray, dose_mci: float) -> np.ndarray:
        """Linearly interpolate frame-mean values as an approximation to continuous DC IDIF."""
        t = np.asarray(relative_time_min, dtype=float)
        if not np.isfinite(t).all() or not np.isfinite(dose_mci) or dose_mci <= 0:
            raise ValueError("time and administered dose must be finite; dose must be positive")
        valid = t >= 0
        if np.any(t[valid] > self.coverage_min + 1e-9):
            raise ValueError(f"{self.tracer} IDIF ends at {self.coverage_min:g} minutes; no extrapolation")
        # Edge-hold inside the measured interval, zero before this tracer's scan/injection.
        values = np.interp(np.clip(t, 0, self.coverage_min), self.mid_min, self.bqml)
        return np.where(valid, values * dose_mci / self.reference_dose_mci, 0.0)


def frame_average(times_min: np.ndarray, values: np.ndarray, edges_min: np.ndarray) -> np.ndarray:
    """Trapezoidal integration on a dense grid, including exact frame endpoints."""
    out = []
    for a, b in zip(edges_min[:-1], edges_min[1:]):
        inside = times_min[(times_min > a) & (times_min < b)]
        points = np.concatenate(([a], inside, [b]))
        out.append(np.trapezoid(np.interp(points, times_min, values), points) / (b - a))
    return np.asarray(out)


def simulate_mixture(
    fdg: MeasuredIDIF,
    pbr: MeasuredIDIF,
    frame_edges_min: np.ndarray,
    *,
    fdg_injection_min: float = 5.0,
    fdg_dose_mci: float = 4.91,
    pbr_dose_mci: float = 4.98,
    time_step_sec: float = 1.0,
    signal_source: str = "tissue-approx",
) -> dict[str, np.ndarray]:
    """Make separate DC targets, undo decay correction, then sum physical activities.

    Timeline zero is PBR28 scan start; FDG injection/scan start is offset
    by fdg_injection_min. We assume each single-tracer scan began at injection.
    Counts/noise are generated separately from the physical mixture.
    """
    edges = np.asarray(frame_edges_min, dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0) or edges[0] < 0):
        raise ValueError("invalid frame edges")
    if not np.isfinite(fdg_injection_min) or fdg_injection_min < 0:
        raise ValueError("FDG offset must be nonnegative")
    if fdg.tracer != "FDG" or pbr.tracer != "PBR28":
        raise ValueError("incorrect IDIF tracer labels")
    if edges[-1] > pbr.coverage_min + 1e-8 or edges[-1] - fdg_injection_min > fdg.coverage_min + 1e-8:
        raise ValueError("requested frames exceed measured IDIF coverage")
    if time_step_sec <= 0 or not np.isfinite(time_step_sec):
        raise ValueError("time_step_sec must be positive")
    if signal_source not in ("blood", "tissue-approx"):
        raise ValueError("signal_source must be 'blood' or 'tissue-approx'")

    # Include every frame edge exactly; helps with short early frames.
    dense = np.unique(np.concatenate((
        np.arange(0, edges[-1], time_step_sec / 60), edges, [edges[-1]]
    )))
    cp_pbr = pbr.decay_corrected_bqml(dense, pbr_dose_mci)
    cp_fdg = fdg.decay_corrected_bqml(dense - fdg_injection_min, fdg_dose_mci)
    if signal_source == "blood":
        dc_pbr, dc_fdg = cp_pbr, cp_fdg
    else:
        # Fixed kinetic coefficients, approximate use of whole-blood IDIF;
        # integrate on a UNIFORM grid to respect Euler's constant dt requirement.
        dt = time_step_sec / 60
        uniform = np.arange(0, edges[-1] + dt / 2, dt)
        if uniform[-1] < edges[-1]:
            uniform = np.append(uniform, edges[-1])
        # Use uniform spacing for the Euler solver, and interpolate to edges later.
        # If the last duration is shorter, remove terminal point (interp edge hold).
        if len(uniform) > 2 and not np.allclose(np.diff(uniform), np.diff(uniform)[0]):
            uniform = np.arange(0, edges[-1] + dt, dt)
        uniform_pbr = pbr.decay_corrected_bqml(np.minimum(uniform, pbr.coverage_min), pbr_dose_mci)
        uniform_fdg = fdg.decay_corrected_bqml(uniform - fdg_injection_min, fdg_dose_mci)
        tissue_pbr, _, _ = simulate_2tcm(**APPROX_TISSUE_PARAMS["PBR28"], Cp=uniform_pbr, t=uniform)
        tissue_fdg, _, _ = simulate_2tcm(**APPROX_TISSUE_PARAMS["FDG"], Cp=uniform_fdg, t=uniform)
        dc_pbr = np.interp(dense, uniform, tissue_pbr)
        dc_fdg = np.interp(dense, uniform, tissue_fdg)
    # Each measured IDIF was decay corrected to its own scan start.
    decay_pbr = np.exp(-np.log(2) * dense / HALF_LIVES_MIN["PBR28"])
    decay_fdg = np.exp(-np.log(2) * np.maximum(dense - fdg_injection_min, 0) / HALF_LIVES_MIN["FDG"])
    phys_pbr = dc_pbr * decay_pbr
    phys_fdg = dc_fdg * decay_fdg
    y_phys = np.stack([frame_average(dense, phys_fdg, edges), frame_average(dense, phys_pbr, edges)])
    y_dc = np.stack([frame_average(dense, dc_fdg, edges), frame_average(dense, dc_pbr, edges)])
    clean = y_phys.sum(axis=0)
    return {
        "Y_phys_tavg_n0_d1": y_phys,
        "Y_bio_tavg_n0_d0": y_dc,
        "X_clean_tavg_n0_d1": clean[np.newaxis, :],
        "fdg_idif_dc_bqml": frame_average(dense, cp_fdg, edges),
        "pbr28_idif_dc_bqml": frame_average(dense, cp_pbr, edges),
    }


def noisy_bqml_from_counts(
    clean_bqml: np.ndarray,
    frame_durations_min: np.ndarray,
    *,
    effective_sensitivity_cps_per_bq: float,
    effective_volume_ml: float = 0.064,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Idealized Poisson count model. Coefficient MUST be calibrated, not scanner sensitivity."""
    concentration = np.asarray(clean_bqml, dtype=float)
    durations = np.asarray(frame_durations_min, dtype=float)
    if (np.any(concentration < 0) or not np.isfinite(concentration).all()
            or np.any(durations <= 0) or not np.isfinite(durations).all()
            or effective_sensitivity_cps_per_bq <= 0 or effective_volume_ml <= 0
            or not np.isfinite(effective_sensitivity_cps_per_bq)):
        raise ValueError("invalid count simulation inputs")
    conversion = durations * 60 * effective_volume_ml * effective_sensitivity_cps_per_bq
    expected = concentration * conversion
    counts = (rng if rng is not None else np.random.default_rng()).poisson(expected)
    return counts / conversion, expected, counts
