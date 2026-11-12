"""Synthetic-only regression tests; no participant-derived input is checked in."""
import csv
import numpy as np
import pytest

from sim.bqml_calibration import MeasuredIDIF, noisy_bqml_from_counts, simulate_mixture
from sim.generate_bqml_dataset import generate_dataset


def make_csv(path, last_sec=120):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TimeStart", "TimeEnd", "TimeMid", "aorta"])
        for a in range(0, last_sec, 5):
            writer.writerow([a, a + 5, a + 2.5, 1000 + 10 * a])


def test_dose_scaling_and_no_extrapolation(tmp_path):
    path = tmp_path / "fdg.csv"
    make_csv(path)
    fdg = MeasuredIDIF.from_csv(path, "FDG")
    t = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(fdg.decay_corrected_bqml(t, 9.82),
                               2 * fdg.decay_corrected_bqml(t, 4.91))
    with pytest.raises(ValueError, match="no extrapolation"):
        fdg.decay_corrected_bqml(np.array([2.01]), 4.91)


def test_physical_sum_and_preinjection_zero(tmp_path):
    a, b = tmp_path / "fdg.csv", tmp_path / "pbr.csv"
    make_csv(a)
    make_csv(b)
    fdg, pbr = MeasuredIDIF.from_csv(a, "FDG"), MeasuredIDIF.from_csv(b, "PBR28")
    result = simulate_mixture(fdg, pbr, np.array([0, 0.5, 1.0, 1.5]),
                              fdg_injection_min=0.5, signal_source="blood")
    y = result["Y_phys_tavg_n0_d1"]
    np.testing.assert_allclose(result["X_clean_tavg_n0_d1"][0], y.sum(axis=0))
    assert y[0, 0] == 0
    assert np.all(y >= 0)
    assert y[1, -1] < result["Y_bio_tavg_n0_d0"][1, -1]


def test_noise_counts_concentration_roundtrip():
    clean = np.array([0, 1000, 5000.0])
    dur = np.array([1 / 12, 0.5, 2.0])
    noisy, expected, sampled = noisy_bqml_from_counts(
        clean, dur, effective_sensitivity_cps_per_bq=0.01,
        rng=np.random.default_rng(123)
    )
    np.testing.assert_allclose(expected, clean * dur * 60 * 0.064 * 0.01)
    np.testing.assert_allclose(noisy, sampled / (dur * 60 * 0.064 * 0.01))
    assert sampled[0] == 0


def test_compatible_dataset(tmp_path):
    a, b = tmp_path / "fdg.csv", tmp_path / "pbr.csv"
    make_csv(a, 180)
    make_csv(b, 180)
    out = tmp_path / "out"
    generate_dataset(a, b, out, n_samples=10, pbr_duration_min=2.0,
                     fdg_offsets_min=(0, 0.5), signal_source="blood")
    with np.load(out / "train.npz") as d:
        assert d["X_measured_tavg_n1_d1"].shape == (8, 1, 24)
        assert d["Y_phys_tavg_n0_d1"].shape == (8, 2, 24)
        np.testing.assert_allclose(d["X_clean_tavg_n0_d1"][:, 0],
                                   d["Y_phys_tavg_n0_d1"].sum(axis=1), atol=1e-3)
