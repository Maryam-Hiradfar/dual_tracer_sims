"""Generate, stack, split, validate, and save a dual-tracer PET dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from core.timegrid import TimeGrid
from sim.dataset_builder import save_dataset_splits, split_dataset, stack_samples
from sim.frame_schedules import default_frame_edges
from sim.sample_generator import DataSetConfig, DatasetGenerator
from tracers.tracer_factory import tracer_factory


def generate_and_save_dataset(
    *,
    output_dir: Path,
    n_samples: int,
    dataset_seed: int,
    split_seed: int,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict:
    """Generate a complete dataset and save deterministic data splits."""
    timegrid = TimeGrid(
        frame_edges=default_frame_edges(),
        internal_dt_min=1 / 60,
    )
    if np.any(timegrid.frame_durs <= 0):
        raise ValueError("all PET frame durations must be positive")

    config = DataSetConfig(
        n_sample=n_samples,
        delta_choices_min=[0.0, 5.0, 10.0],
        scale_range=(0.85, 1.15),
        first_tracer_name="PBR28",
        second_tracer_name="FDG",
        randomize_order=False,
    )
    generator = DatasetGenerator(
        timegrid=timegrid,
        tracer_factory=tracer_factory,
        config=config,
        seed=dataset_seed,
    )
    samples, specs = generator.generate_dataset()
    arrays_and_metadata = stack_samples(samples, specs)
    splits = split_dataset(
        *arrays_and_metadata,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=split_seed,
    )

    fdg = tracer_factory("FDG")
    pbr28 = tracer_factory("PBR28")
    global_metadata = {
        "n_samples": n_samples,
        "dataset_seed": dataset_seed,
        "split_seed": split_seed,
        "split_fractions": {
            "train": train_fraction,
            "val": val_fraction,
            "test": test_fraction,
        },
        "dataset_config": {
            "delta_choices_min": config.delta_choices_min,
            "scale_range": config.scale_range,
            "first_tracer_name": config.first_tracer_name,
            "second_tracer_name": config.second_tracer_name,
            "randomize_order": config.randomize_order,
        },
        "frame_edges_min": timegrid.frame_edges,
        "frame_midpoints_min": timegrid.frame_mids,
        "frame_durations_min": timegrid.frame_durs,
        "internal_dt_min": timegrid.internal_dt_min,
        "scanner_name": "panorama_gs",
        "voxel_size_mm": 4.0,
        "noise_model": "Poisson counts",
        "channel_definitions": {
            "input_channel_0": "noisy mixed FDG + PBR28 measured-domain TAC",
            "target_channel_0": "FDG",
            "target_channel_1": "PBR28",
        },
        "units": {
            "time": "minutes",
            "TAC": "kBq/mL",
            "expected_counts": "counts per frame",
        },
        "fixed_kinetic_parameters": {
            "FDG": fdg.true_params(),
            "PBR28": pbr28.true_params(),
        },
        "tracer_half_lives_min": {
            "FDG": fdg.half_life_min,
            "PBR28": pbr28.half_life_min,
        },
    }
    save_dataset_splits(output_dir, splits, global_metadata)
    return {
        "output_dir": Path(output_dir),
        "split_sizes": {
            name: len(split["sample_indices"])
            for name, split in splits.items()
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate train, validation, and test PET simulation files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/dual_tracer"),
    )
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--dataset-seed", type=int, default=48)
    parser.add_argument("--split-seed", type=int, default=49)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive")

    summary = generate_and_save_dataset(
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        dataset_seed=args.dataset_seed,
        split_seed=args.split_seed,
    )
    print(f"Saved dataset to {summary['output_dir']}")
    print(f"Split sizes: {summary['split_sizes']}")


if __name__ == "__main__":
    main()
