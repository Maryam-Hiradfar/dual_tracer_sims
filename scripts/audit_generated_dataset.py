"""Audit a saved dual-tracer dataset and plot representative samples."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from sim.dataset_builder import ARRAY_NAMES
from sim.plotters import plot_representative_dataset_samples


def _array_stats(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "percentiles": {
            str(percentile): float(np.percentile(array, percentile))
            for percentile in (1, 5, 50, 95, 99)
        },
    }


def _tracer_scale(item: dict[str, Any], tracer_name: str) -> float:
    if item["tracer1_name"] == tracer_name:
        return float(item["tracer1_scale"])
    if item["tracer2_name"] == tracer_name:
        return float(item["tracer2_scale"])
    raise ValueError(f"sample {item.get('sample_id')} does not contain {tracer_name}")


def _scale_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def _select_representatives(
    arrays: dict[str, np.ndarray],
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Choose low, balanced, and high FDG/PBR28 scale-ratio examples."""
    selections = []
    strategies = {
        0.0: "low FDG/PBR28 ratio",
        5.0: "balanced FDG/PBR28 ratio",
        10.0: "high FDG/PBR28 ratio",
    }
    for delay, strategy in strategies.items():
        candidates = [
            index
            for index, item in enumerate(metadata)
            if float(item["delta_min"]) == delay
        ]
        if not candidates:
            raise ValueError(f"no sample found for delay {delay}")

        ratios = np.asarray([
            _tracer_scale(metadata[index], "FDG")
            / _tracer_scale(metadata[index], "PBR28")
            for index in candidates
        ])
        if delay == 0.0:
            chosen = candidates[int(np.argmin(ratios))]
        elif delay == 5.0:
            chosen = candidates[int(np.argmin(np.abs(ratios - 1.0)))]
        else:
            chosen = candidates[int(np.argmax(ratios))]

        item = metadata[chosen]
        fdg_scale = _tracer_scale(item, "FDG")
        pbr_scale = _tracer_scale(item, "PBR28")
        selections.append({
            "sample_id": int(item["sample_id"]),
            "measured": arrays["X_measured_tavg_n1_d1"][chosen, 0],
            "clean": arrays["X_clean_tavg_n0_d1"][chosen, 0],
            "fdg_phys": arrays["Y_phys_tavg_n0_d1"][chosen, 0],
            "pbr_phys": arrays["Y_phys_tavg_n0_d1"][chosen, 1],
            "fdg_bio": arrays["Y_bio_tavg_n0_d0"][chosen, 0],
            "pbr_bio": arrays["Y_bio_tavg_n0_d0"][chosen, 1],
            "title": (
                f"delay={delay:g} min | FDG={fdg_scale:.3f}, "
                f"PBR28={pbr_scale:.3f}\n{strategy} | sample {item['sample_id']}"
            ),
            "selection_reason": strategy,
            "delay_min": delay,
            "fdg_scale": fdg_scale,
            "pbr28_scale": pbr_scale,
        })
    return selections


def audit_dataset(data_dir: Path) -> dict[str, Any]:
    global_metadata = json.loads((data_dir / "dataset_metadata.json").read_text())
    split_arrays = {}
    split_metadata = {}

    for split_name in ("train", "val", "test"):
        with np.load(data_dir / f"{split_name}.npz") as data:
            split_arrays[split_name] = {
                name: data[name].copy() for name in ARRAY_NAMES
            }
            split_arrays[split_name]["sample_indices"] = data[
                "sample_indices"
            ].copy()
        split_metadata[split_name] = json.loads(
            (data_dir / f"{split_name}_metadata.json").read_text()
        )

    arrays = {
        name: np.concatenate(
            [split_arrays[split][name] for split in ("train", "val", "test")]
        )
        for name in ARRAY_NAMES
    }
    metadata = sum(
        [split_metadata[split] for split in ("train", "val", "test")],
        [],
    )
    sample_indices = np.concatenate([
        split_arrays[split]["sample_indices"]
        for split in ("train", "val", "test")
    ])
    n_samples = len(metadata)

    checks = {
        "sample_count_matches_global_metadata": (
            n_samples == int(global_metadata["n_samples"])
        ),
        "sample_indices_complete_and_unique": bool(
            np.array_equal(np.sort(sample_indices), np.arange(n_samples))
        ),
        "sample_ids_complete_and_unique": (
            sorted(int(item["sample_id"]) for item in metadata)
            == list(range(n_samples))
        ),
        "sample_seeds_present_and_unique": (
            len({int(item["seed"]) for item in metadata}) == n_samples
        ),
        "all_arrays_finite": all(np.isfinite(array).all() for array in arrays.values()),
        "clean_and_target_arrays_nonnegative": all(
            np.all(arrays[name] >= 0)
            for name in (
                "X_clean_tavg_n0_d1",
                "Y_phys_tavg_n0_d1",
                "Y_bio_tavg_n0_d0",
                "expected_counts",
            )
        ),
        "clean_mixture_equals_physical_sum": bool(
            np.allclose(
                arrays["X_clean_tavg_n0_d1"][:, 0],
                arrays["Y_phys_tavg_n0_d1"].sum(axis=1),
                rtol=1e-5,
                atol=1e-5,
            )
        ),
        "fixed_pbr28_then_fdg_order": all(
            item["tracer1_name"] == "PBR28"
            and item["tracer2_name"] == "FDG"
            for item in metadata
        ),
        "all_ml_arrays_float32": all(
            array.dtype == np.float32 for array in arrays.values()
        ),
    }

    fdg_scales = [_tracer_scale(item, "FDG") for item in metadata]
    pbr_scales = [_tracer_scale(item, "PBR28") for item in metadata]
    delay_counts = Counter(float(item["delta_min"]) for item in metadata)
    split_delay_counts = {
        split: dict(sorted(Counter(
            float(item["delta_min"]) for item in split_metadata[split]
        ).items()))
        for split in ("train", "val", "test")
    }

    clean = arrays["X_clean_tavg_n0_d1"]
    noise = arrays["X_measured_tavg_n1_d1"] - clean
    noise_mse = float(np.mean(noise**2))
    clean_power = float(np.mean(clean**2))
    snr_db = float(10 * np.log10(clean_power / noise_mse))

    representatives = _select_representatives(arrays, metadata)
    plot_paths = plot_representative_dataset_samples(
        representatives,
        np.asarray(global_metadata["frame_midpoints_min"]),
        data_dir / "audit_plots",
    )

    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "configuration": global_metadata,
        "actual_sample_count": n_samples,
        "actual_split_sizes": {
            split: len(split_metadata[split])
            for split in ("train", "val", "test")
        },
        "actual_parameter_statistics": {
            "FDG_scale": _scale_stats(fdg_scales),
            "PBR28_scale": _scale_stats(pbr_scales),
            "delay_counts": dict(sorted(delay_counts.items())),
            "delay_counts_by_split": split_delay_counts,
            "injection_order_counts": dict(Counter(
                f"{item['tracer1_name']} then {item['tracer2_name']}"
                for item in metadata
            )),
        },
        "array_statistics": {
            name: _array_stats(array) for name, array in arrays.items()
        },
        "noise_statistics": {
            "noise_mse_kBq2_per_mL2": noise_mse,
            "clean_signal_power_kBq2_per_mL2": clean_power,
            "global_snr_db": snr_db,
        },
        "representative_samples": [
            {
                key: value
                for key, value in sample.items()
                if not isinstance(value, np.ndarray)
            }
            for sample in representatives
        ],
        "plot_files": [str(path) for path in plot_paths],
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/dual_tracer"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_dataset(args.data_dir)
    report_path = args.data_dir / "audit_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Dataset audit: {report['status']}")
    print(f"Samples: {report['actual_sample_count']}")
    print(f"Splits: {report['actual_split_sizes']}")
    print(f"FDG scale: {report['actual_parameter_statistics']['FDG_scale']}")
    print(f"PBR28 scale: {report['actual_parameter_statistics']['PBR28_scale']}")
    print(f"Delay counts: {report['actual_parameter_statistics']['delay_counts']}")
    print(f"Global SNR: {report['noise_statistics']['global_snr_db']:.3f} dB")
    print(f"Report: {report_path}")
    for plot_file in report["plot_files"]:
        print(f"Plot: {plot_file}")

    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
