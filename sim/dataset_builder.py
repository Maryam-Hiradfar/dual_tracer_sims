"""Canonicalize and stack simulated dual-tracer samples."""

import json
from pathlib import Path
from typing import Any

import numpy as np


ARRAY_NAMES = (
    "X_measured_tavg_n1_d1",
    "X_clean_tavg_n0_d1",
    "Y_phys_tavg_n0_d1",
    "Y_bio_tavg_n0_d0",
    "expected_counts",
)


def canonicalize_sample(sample, spec):
    """Return one sample with FDG first and PBR28 second."""
    if spec.tracer1_name == "FDG" and spec.tracer2_name == "PBR28":
        fdg_bio, pbr28_bio = sample.y_tracer1_true_tavg_n0_d0, sample.y_tracer2_true_tavg_n0_d0
        fdg_phys, pbr28_phys = sample.y_tracer1_phys_tavg_n0_d1, sample.y_tracer2_phys_tavg_n0_d1
    elif spec.tracer1_name == "PBR28" and spec.tracer2_name == "FDG":
        fdg_bio, pbr28_bio = sample.y_tracer2_true_tavg_n0_d0, sample.y_tracer1_true_tavg_n0_d0
        fdg_phys, pbr28_phys = sample.y_tracer2_phys_tavg_n0_d1, sample.y_tracer1_phys_tavg_n0_d1
    else:
        raise ValueError(
            f"Unexpected tracer order: {spec.tracer1_name}, {spec.tracer2_name}"
        )

    return {
        "X_measured_tavg_n1_d1": sample.y_meas_tavg_n1_d1,
        "X_clean_tavg_n0_d1": sample.y_clean_tavg_n0_d1,
        "Y_phys_tavg_n0_d1": (fdg_phys, pbr28_phys),
        "Y_bio_tavg_n0_d0": (fdg_bio, pbr28_bio),
        "expected_counts": sample.expected_counts,
        "metadata": {
            **sample.metadata,
            "tracer1_name": spec.tracer1_name,
            "tracer2_name": spec.tracer2_name,
            "tracer1_scale": spec.tracer1_scale,
            "tracer2_scale": spec.tracer2_scale,
            "delta_min": spec.delta_min,
            "tracer1_first": spec.tracer1_first,
            "seed": spec.seed,
        },
    }


def stack_samples(samples, specs):
    """Stack samples into validated, canonical, float32 dataset arrays."""
    if len(samples) != len(specs):
        raise ValueError(
            "samples and specs must have equal length: "
            f"received {len(samples)} samples and {len(specs)} specs"
        )
    if not samples:
        raise ValueError("samples and specs must not be empty")

    canonical = [
        canonicalize_sample(sample, spec)
        for sample, spec in zip(samples, specs)
    ]

    frame_arrays = []
    for sample_index, item in enumerate(canonical):
        arrays = (
            item["X_measured_tavg_n1_d1"],
            item["X_clean_tavg_n0_d1"],
            *item["Y_phys_tavg_n0_d1"],
            *item["Y_bio_tavg_n0_d0"],
            item["expected_counts"],
        )
        for array in arrays:
            value = np.asarray(array)
            if value.ndim != 1:
                raise ValueError(
                    f"all TAC arrays must be one-dimensional; sample "
                    f"{sample_index} has shape {value.shape}"
                )
            frame_arrays.append((sample_index, value))

    frame_length = frame_arrays[0][1].shape[0]
    for sample_index, array in frame_arrays:
        if array.shape[0] != frame_length:
            raise ValueError(
                "all TAC arrays must have the same frame length; "
                f"expected {frame_length}, but sample {sample_index} has "
                f"length {array.shape[0]}"
            )

    X_measured_tavg_n1_d1 = np.stack(
        [item["X_measured_tavg_n1_d1"] for item in canonical]
    )[:, None, :].astype(np.float32)
    X_clean_tavg_n0_d1 = np.stack(
        [item["X_clean_tavg_n0_d1"] for item in canonical]
    )[:, None, :].astype(np.float32)
    Y_phys_tavg_n0_d1 = np.stack(
        [np.stack(item["Y_phys_tavg_n0_d1"], axis=0) for item in canonical]
    ).astype(np.float32)
    Y_bio_tavg_n0_d0 = np.stack(
        [np.stack(item["Y_bio_tavg_n0_d0"], axis=0) for item in canonical]
    ).astype(np.float32)
    expected_counts = np.stack(
        [item["expected_counts"] for item in canonical]
    ).astype(np.float32)

    output_arrays = (X_measured_tavg_n1_d1, X_clean_tavg_n0_d1, Y_phys_tavg_n0_d1, Y_bio_tavg_n0_d0, expected_counts)
    if any(not np.isfinite(array).all() for array in output_arrays):
        raise ValueError("dataset arrays must not contain NaN or infinite values")

    physical_sum = Y_phys_tavg_n0_d1[:, 0, :] + Y_phys_tavg_n0_d1[:, 1, :]
    if not np.allclose(X_clean_tavg_n0_d1[:, 0, :], physical_sum):
        raise ValueError("X_clean_tavg_n0_d1 does not equal the sum of the physical TACs")

    metadata = [
        {**item["metadata"], "sample_id": sample_id}
        for sample_id, item in enumerate(canonical)
    ]
    return X_measured_tavg_n1_d1, X_clean_tavg_n0_d1, Y_phys_tavg_n0_d1, Y_bio_tavg_n0_d0, expected_counts, metadata


def split_dataset(
    X_measured_tavg_n1_d1,
    X_clean_tavg_n0_d1,
    Y_phys_tavg_n0_d1,
    Y_bio_tavg_n0_d0,
    expected_counts,
    metadata,
    *,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
):
    """Split canonical arrays into deterministic train, validation, and test sets."""
    fractions = np.asarray(
        [train_fraction, val_fraction, test_fraction],
        dtype=float,
    )
    if np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("train, validation, and test fractions must be non-negative and sum to 1")

    arrays = {
        "X_measured_tavg_n1_d1": np.asarray(X_measured_tavg_n1_d1),
        "X_clean_tavg_n0_d1": np.asarray(X_clean_tavg_n0_d1),
        "Y_phys_tavg_n0_d1": np.asarray(Y_phys_tavg_n0_d1),
        "Y_bio_tavg_n0_d0": np.asarray(Y_bio_tavg_n0_d0),
        "expected_counts": np.asarray(expected_counts),
    }
    n_samples = arrays["X_measured_tavg_n1_d1"].shape[0]
    if len(metadata) != n_samples:
        raise ValueError(
            f"metadata has {len(metadata)} entries, but arrays have {n_samples} samples"
        )
    for name, array in arrays.items():
        if array.ndim == 0 or array.shape[0] != n_samples:
            raise ValueError(
                f"{name} must have {n_samples} samples on axis 0; received {array.shape}"
            )

    shuffled_indices = np.random.default_rng(seed).permutation(n_samples)
    n_train = int(train_fraction * n_samples)
    n_val = int(val_fraction * n_samples)
    split_indices = {
        "train": shuffled_indices[:n_train],
        "val": shuffled_indices[n_train:n_train + n_val],
        "test": shuffled_indices[n_train + n_val:],
    }
    _validate_split_indices(split_indices, n_samples)

    return {
        split_name: {
            **{name: array[indices] for name, array in arrays.items()},
            "sample_indices": indices.astype(np.int64),
            "metadata": [metadata[int(index)] for index in indices],
        }
        for split_name, indices in split_indices.items()
    }


def _validate_split_indices(split_indices, n_samples: int) -> None:
    """Verify that split indices are disjoint and cover every sample once."""
    all_indices = np.concatenate(list(split_indices.values()))
    if len(all_indices) != n_samples:
        raise ValueError("split indices do not assign every sample")
    if len(np.unique(all_indices)) != n_samples:
        raise ValueError("split indices overlap")
    if not np.array_equal(np.sort(all_indices), np.arange(n_samples)):
        raise ValueError("split indices are incomplete or outside the dataset")


def _json_safe(value: Any) -> Any:
    """Recursively convert NumPy and Path values into JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def save_dataset_splits(output_dir, splits, global_metadata) -> None:
    """Save one compressed NPZ and one metadata JSON file per split."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_sizes = {}
    for split_name in ("train", "val", "test"):
        if split_name not in splits:
            raise ValueError(f"missing required split: {split_name}")
        split = splits[split_name]
        missing = [name for name in ARRAY_NAMES if name not in split]
        if missing:
            raise ValueError(f"{split_name} split is missing arrays: {missing}")

        np.savez_compressed(
            output_dir / f"{split_name}.npz",
            **{name: split[name] for name in ARRAY_NAMES},
            sample_indices=np.asarray(split["sample_indices"], dtype=np.int64),
        )
        with (output_dir / f"{split_name}_metadata.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(_json_safe(split["metadata"]), file, indent=2)
            file.write("\n")
        split_sizes[split_name] = len(split["sample_indices"])

    dataset_metadata = {
        **global_metadata,
        "split_sizes": split_sizes,
    }
    with (output_dir / "dataset_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(_json_safe(dataset_metadata), file, indent=2)
        file.write("\n")


def load_dataset_split(output_dir, split_name):
    """Load one saved split and its per-sample metadata."""
    if split_name not in {"train", "val", "test"}:
        raise ValueError("split_name must be 'train', 'val', or 'test'")

    output_dir = Path(output_dir)
    with np.load(output_dir / f"{split_name}.npz") as data:
        split = {name: data[name].copy() for name in ARRAY_NAMES}
        split["sample_indices"] = data["sample_indices"].copy()
    with (output_dir / f"{split_name}_metadata.json").open(
        "r", encoding="utf-8"
    ) as file:
        split["metadata"] = json.load(file)
    return split


def load_dataset_metadata(output_dir):
    """Load global dataset metadata."""
    with (Path(output_dir) / "dataset_metadata.json").open(
        "r", encoding="utf-8"
    ) as file:
        return json.load(file)
