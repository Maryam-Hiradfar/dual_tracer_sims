import numpy as np
import pytest

from core.protocol import Protocol
from core.timegrid import TimeGrid
from sim.sample_generator import DataSetConfig, DatasetGenerator, SampleGenerator
from sim.tac_simulator import (
    estimate_biological_curves,
    simulate_dual_tracer_measurement,
)
from tracers.fdg import FDGTracer
from tracers.pbr28 import PBR28Tracer
from tracers.tracer_factory import tracer_factory
from utils.frame_average import frame_average


def _simulation_inputs():
    timegrid = TimeGrid(
        frame_edges=np.array([0.0, 0.25, 0.5, 1.0, 2.0]),
        internal_dt_min=1 / 60,
    )
    fdg = FDGTracer(name="FDG", half_life_min=109.8, scale=1.0)
    pbr28 = PBR28Tracer(name="PBR28", half_life_min=20.4, scale=1.0)
    protocol = Protocol(delta_min=0.25)
    return timegrid, fdg, pbr28, protocol


def test_frame_clean_tac_is_sum_of_frame_physical_tacs():
    timegrid, fdg, pbr28, protocol = _simulation_inputs()

    sim = simulate_dual_tracer_measurement(
        timegrid=timegrid,
        tracer1=fdg,
        tracer2=pbr28,
        protocol=protocol,
        rng=np.random.default_rng(7),
    )

    np.testing.assert_allclose(sim.y_clean_tavg_n0_d1, sim.Ct1_phys_tavg_n0_d1 + sim.Ct2_phys_tavg_n0_d1)
    np.testing.assert_allclose(
        sim.y_clean_tint_n0_d1,
        sim.Ct1_phys_tint_n0_d1 + sim.Ct2_phys_tint_n0_d1,
    )
    assert sim.y_clean_tavg_n0_d1.shape == timegrid.frame_mids.shape
    assert sim.expected_counts.shape == timegrid.frame_mids.shape


def test_hybrid_sample_retains_all_frame_level_arrays():
    timegrid, fdg, pbr28, protocol = _simulation_inputs()
    generator = SampleGenerator(
        timegrid=timegrid,
        tracer1=fdg,
        tracer2=pbr28,
        protocol=protocol,
        rng=np.random.default_rng(11),
    )

    sample = generator.generate_sample()

    retained_arrays = (
        sample.y_meas_tavg_n1_d1,
        sample.y_clean_tavg_n0_d1,
        sample.y_tracer1_true_tavg_n0_d0,
        sample.y_tracer2_true_tavg_n0_d0,
        sample.y_tracer1_phys_tavg_n0_d1,
        sample.y_tracer2_phys_tavg_n0_d1,
        sample.expected_counts,
    )
    assert all(array.shape == timegrid.frame_mids.shape for array in retained_arrays)
    np.testing.assert_allclose(
        sample.y_clean_tavg_n0_d1,
        sample.y_tracer1_phys_tavg_n0_d1 + sample.y_tracer2_phys_tavg_n0_d1,
    )


def test_dataset_seed_reproduces_specs_and_noise_with_child_seeds():
    timegrid, _, _, _ = _simulation_inputs()
    config = DataSetConfig(
        n_sample=2,
        delta_choices_min=[0.0, 0.25],
        scale_range=(0.8, 1.2),
        randomize_order=True,
    )

    first = DatasetGenerator(timegrid, tracer_factory, config, seed=23)
    second = DatasetGenerator(timegrid, tracer_factory, config, seed=23)
    first_samples, first_specs = first.generate_dataset()
    second_samples, second_specs = second.generate_dataset()

    assert first_specs == second_specs
    assert all(spec.seed is not None for spec in first_specs)
    for first_sample, second_sample in zip(first_samples, second_samples):
        np.testing.assert_array_equal(
            first_sample.y_meas_tavg_n1_d1,
            second_sample.y_meas_tavg_n1_d1,
        )


def test_different_dataset_seed_changes_generated_specs():
    timegrid, _, _, _ = _simulation_inputs()
    config = DataSetConfig(
        n_sample=2,
        delta_choices_min=[0.0, 0.25],
        scale_range=(0.8, 1.2),
        randomize_order=True,
    )

    _, first_specs = DatasetGenerator(
        timegrid, tracer_factory, config, seed=23
    ).generate_dataset()
    _, second_specs = DatasetGenerator(
        timegrid, tracer_factory, config, seed=24
    ).generate_dataset()

    assert first_specs != second_specs


def test_fixed_tracer_order_keeps_pbr28_first_and_fdg_second():
    timegrid, _, _, _ = _simulation_inputs()
    config = DataSetConfig(
        n_sample=5,
        delta_choices_min=[0.0, 0.25],
        scale_range=(0.8, 1.2),
        first_tracer_name="PBR28",
        second_tracer_name="FDG",
        randomize_order=False,
    )

    _, specs = DatasetGenerator(
        timegrid, tracer_factory, config, seed=31
    ).generate_dataset()

    assert all(spec.tracer1_name == "PBR28" for spec in specs)
    assert all(spec.tracer2_name == "FDG" for spec in specs)
    assert all(spec.tracer1_first for spec in specs)


def test_pbr28_aif_uses_parameter_overrides():
    default = PBR28Tracer(name="PBR28", half_life_min=20.4)
    overridden = PBR28Tracer(
        name="PBR28",
        half_life_min=20.4,
        pbr28_params={"A1": 44.0},
    )
    time = np.array([0.0, 1.0])

    assert not np.array_equal(default.aif(time), overridden.aif(time))
    assert overridden.aif(time)[0] == pytest.approx(47.45)


def test_frame_average_integrates_linear_curve_exactly():
    time = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    values = 2.0 * time + 1.0

    averaged = frame_average(time, values, np.array([0.0, 1.0, 2.0]))

    np.testing.assert_allclose(averaged, np.array([2.0, 4.0]))


@pytest.mark.parametrize(
    "frame_edges, internal_dt",
    [
        ([0.0], 1 / 60),
        ([0.0, 1.0, 0.5], 1 / 60),
        ([0.0, np.nan, 1.0], 1 / 60),
        ([0.0, 1.0], 0.0),
        ([0.0, 0.01], 1 / 60),
    ],
)
def test_timegrid_rejects_invalid_configuration(frame_edges, internal_dt):
    with pytest.raises(ValueError):
        TimeGrid(np.asarray(frame_edges), internal_dt_min=internal_dt)


def test_decay_correction_uses_frame_integrated_decay_factors():
    timegrid, fdg, pbr28, protocol = _simulation_inputs()
    sim = simulate_dual_tracer_measurement(
        timegrid=timegrid,
        tracer1=fdg,
        tracer2=pbr28,
        protocol=protocol,
        rng=np.random.default_rng(3),
    )

    expected_decay1 = frame_average(
        sim.t_int,
        np.exp(-sim.lam1 * sim.t_int),
        timegrid.frame_edges,
    )
    expected_decay2 = frame_average(
        sim.t_int,
        np.exp(-sim.lam2 * np.maximum(sim.t_int - sim.delta_min, 0.0)),
        timegrid.frame_edges,
    )
    np.testing.assert_allclose(sim.decay1_tavg, expected_decay1)
    np.testing.assert_allclose(sim.decay2_tavg, expected_decay2)

    tracer1_bio, tracer2_bio = estimate_biological_curves(
        sim.Ct1_phys_tavg_n0_d1,
        sim.Ct2_phys_tavg_n0_d1,
        sim.decay1_tavg,
        sim.decay2_tavg,
    )
    assert np.isfinite(tracer1_bio).all()
    assert np.isfinite(tracer2_bio).all()
