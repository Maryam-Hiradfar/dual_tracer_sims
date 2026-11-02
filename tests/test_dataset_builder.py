from types import SimpleNamespace

import numpy as np

from sim.dataset_builder import (
    ARRAY_NAMES,
    load_dataset_metadata,
    load_dataset_split,
    save_dataset_splits,
    split_dataset,
    stack_samples,
)


def _sample(tracer1_value, tracer2_value, frame_count=4):
    tracer1_bio = np.full(frame_count, tracer1_value, dtype=np.float64)
    tracer2_bio = np.full(frame_count, tracer2_value, dtype=np.float64)
    tracer1_phys = tracer1_bio * 0.5
    tracer2_phys = tracer2_bio * 0.5
    return SimpleNamespace(
        y_meas_tavg_n1_d1=tracer1_phys + tracer2_phys + 0.1,
        y_clean_tavg_n0_d1=tracer1_phys + tracer2_phys,
        y_tracer1_true_tavg_n0_d0=tracer1_bio,
        y_tracer2_true_tavg_n0_d0=tracer2_bio,
        y_tracer1_phys_tavg_n0_d1=tracer1_phys,
        y_tracer2_phys_tavg_n0_d1=tracer2_phys,
        expected_counts=np.full(frame_count, 100.0),
        metadata={"sample_id": 0},
    )


def _spec(tracer1_name, tracer2_name):
    return SimpleNamespace(
        tracer1_name=tracer1_name,
        tracer2_name=tracer2_name,
        tracer1_scale=1.0,
        tracer2_scale=1.0,
        delta_min=0.0,
        tracer1_first=True,
        seed=7,
    )


def test_stack_samples_uses_canonical_fdg_pbr28_channel_order():
    samples = [_sample(2.0, 8.0), _sample(8.0, 2.0)]
    specs = [_spec("FDG", "PBR28"), _spec("PBR28", "FDG")]

    _, _, Y_phys_tavg_n0_d1, Y_bio_tavg_n0_d0, _, _ = stack_samples(samples, specs)

    np.testing.assert_array_equal(Y_bio_tavg_n0_d0[:, 0, :], 2.0)
    np.testing.assert_array_equal(Y_bio_tavg_n0_d0[:, 1, :], 8.0)
    np.testing.assert_array_equal(Y_phys_tavg_n0_d1[:, 0, :], 1.0)
    np.testing.assert_array_equal(Y_phys_tavg_n0_d1[:, 1, :], 4.0)


def test_stack_samples_returns_required_shapes_and_float32_arrays():
    outputs = stack_samples(
        [_sample(2.0, 8.0), _sample(8.0, 2.0)],
        [_spec("FDG", "PBR28"), _spec("PBR28", "FDG")],
    )
    X_measured_tavg_n1_d1, X_clean_tavg_n0_d1, Y_phys_tavg_n0_d1, Y_bio_tavg_n0_d0, expected_counts, metadata = outputs

    assert X_measured_tavg_n1_d1.shape == (2, 1, 4)
    assert X_clean_tavg_n0_d1.shape == (2, 1, 4)
    assert Y_phys_tavg_n0_d1.shape == (2, 2, 4)
    assert Y_bio_tavg_n0_d0.shape == (2, 2, 4)
    assert expected_counts.shape == (2, 4)
    assert all(array.dtype == np.float32 for array in outputs[:5])
    assert len(metadata) == 2


def test_stack_samples_reconstructs_clean_mixture():
    _, X_clean_tavg_n0_d1, Y_phys_tavg_n0_d1, _, _, _ = stack_samples(
        [_sample(2.0, 8.0)],
        [_spec("FDG", "PBR28")],
    )

    np.testing.assert_allclose(
        X_clean_tavg_n0_d1[:, 0, :],
        Y_phys_tavg_n0_d1[:, 0, :] + Y_phys_tavg_n0_d1[:, 1, :],
    )


def _stacked_dataset(sample_count=10):
    samples = [_sample(2.0, 8.0) for _ in range(sample_count)]
    specs = [_spec("FDG", "PBR28") for _ in range(sample_count)]
    return stack_samples(samples, specs)


def test_split_dataset_is_complete_disjoint_and_deterministic():
    dataset = _stacked_dataset()

    first = split_dataset(*dataset, seed=41)
    second = split_dataset(*dataset, seed=41)

    all_indices = np.concatenate(
        [first[name]["sample_indices"] for name in ("train", "val", "test")]
    )
    assert len(np.unique(all_indices)) == 10
    np.testing.assert_array_equal(np.sort(all_indices), np.arange(10))
    assert {name: len(first[name]["sample_indices"]) for name in first} == {
        "train": 8,
        "val": 1,
        "test": 1,
    }
    for name in ("train", "val", "test"):
        np.testing.assert_array_equal(
            first[name]["sample_indices"],
            second[name]["sample_indices"],
        )


def test_save_load_round_trip_preserves_arrays_and_metadata(tmp_path):
    splits = split_dataset(*_stacked_dataset(), seed=7)
    global_metadata = {
        "dataset_seed": np.int64(48),
        "frame_edges_min": np.array([0.0, 1.0, 2.0]),
    }

    save_dataset_splits(tmp_path, splits, global_metadata)

    for split_name in ("train", "val", "test"):
        loaded = load_dataset_split(tmp_path, split_name)
        for array_name in (*ARRAY_NAMES, "sample_indices"):
            np.testing.assert_array_equal(
                loaded[array_name],
                splits[split_name][array_name],
            )
        assert loaded["metadata"] == splits[split_name]["metadata"]

    loaded_global_metadata = load_dataset_metadata(tmp_path)
    assert loaded_global_metadata["dataset_seed"] == 48
    assert loaded_global_metadata["frame_edges_min"] == [0.0, 1.0, 2.0]
    assert loaded_global_metadata["split_sizes"] == {
        "train": 8,
        "val": 1,
        "test": 1,
    }
