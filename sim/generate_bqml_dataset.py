"""Generate TCN-compatible Bq/mL datasets using locally held measured aortic IDIFs.

Example (from repository root):
python -m sim.generate_bqml_dataset --fdg-csv '/path/FDG Aorta(1).csv' \
  --pbr-csv '/path/Aorta PBR(1).csv' --output-dir datasets/bqml_v1 \
  --n-samples 1000 --effective-sensitivity-cps-per-bq 0.001

The effective sensitivity is an illustrative placeholder unless calibrated against
reconstructed data; it is NOT the manufacturer's whole-scanner sensitivity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim.bqml_calibration import (
    MCI_TO_MBQ, MeasuredIDIF, noisy_bqml_from_counts, simulate_mixture
)


def generate_dataset(
    fdg_csv: Path,
    pbr_csv: Path,
    output_dir: Path,
    *,
    n_samples: int = 1000,
    seed: int = 48,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    pbr_duration_min: float = 65.0,
    fdg_offsets_min: tuple[float, ...] = (0, 5, 10),
    dose_relative_range: tuple[float, float] = (0.85, 1.15),
    effective_sensitivity_cps_per_bq: float = 0.001,
    effective_volume_ml: float = 0.064,
    signal_source: str = "tissue-approx",
    fdg_kinetics: dict[str, float] | None = None,
    fdg_vb: float = 0.0,
    fdg_delay_sec: float = 0.0,
) -> None:
    fdg = MeasuredIDIF.from_csv(fdg_csv, "FDG")
    pbr = MeasuredIDIF.from_csv(pbr_csv, "PBR28")
    if n_samples < 3 or not (0 < train_fraction < 1 and 0 < val_fraction < 1
                             and train_fraction + val_fraction < 1):
        raise ValueError("need >=3 samples and nonempty train/validation/test fractions")
    if any(d < 0 for d in fdg_offsets_min) or not fdg_offsets_min:
        raise ValueError("offsets must be nonempty and nonnegative")
    if dose_relative_range[0] <= 0 or dose_relative_range[1] < dose_relative_range[0]:
        raise ValueError("invalid dose range")
    # Use PBR28's actual frame boundaries, truncate at chosen endpoint.
    # FDG has only 65 minutes of measured data; do not extrapolate either IDIF.
    if (pbr_duration_min > pbr.coverage_min or
            pbr_duration_min > fdg.coverage_min + min(fdg_offsets_min)):
        raise ValueError("duration exceeds one of the measured IDIFs at the minimum FDG offset")
    edges = np.r_[pbr.start_min[pbr.start_min < pbr_duration_min], pbr_duration_min]
    if edges[-1] <= edges[0]:
        raise ValueError("no valid frames")
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n_samples):
        offset = float(rng.choice(fdg_offsets_min))
        fdg_dose = float(fdg.reference_dose_mci * rng.uniform(*dose_relative_range))
        pbr_dose = float(pbr.reference_dose_mci * rng.uniform(*dose_relative_range))
        sample = simulate_mixture(
            fdg, pbr, edges, fdg_injection_min=offset, fdg_dose_mci=fdg_dose,
            pbr_dose_mci=pbr_dose, signal_source=signal_source,
            fdg_kinetics=fdg_kinetics, fdg_vb=fdg_vb, fdg_delay_sec=fdg_delay_sec
        )
        observed, expected, counts = noisy_bqml_from_counts(
            sample["X_clean_tavg_n0_d1"][0], np.diff(edges),
            effective_sensitivity_cps_per_bq=effective_sensitivity_cps_per_bq,
            effective_volume_ml=effective_volume_ml, rng=rng
        )
        samples.append({
            **sample,
            "X_measured_tavg_n1_d1": observed[None, :],
            "expected_counts": expected,
            "sampled_counts": counts,
            "metadata": {
                "sample_id": i, "delta_min": offset,
                "fdg_dose_mci": fdg_dose, "pbr28_dose_mci": pbr_dose,
                "fdg_dose_mbq": fdg_dose * MCI_TO_MBQ,
                "pbr28_dose_mbq": pbr_dose * MCI_TO_MBQ,
                "tracer1_name": "PBR28", "tracer2_name": "FDG",
                "tracer1_scale": pbr_dose / pbr.reference_dose_mci,
                "tracer2_scale": fdg_dose / fdg.reference_dose_mci,
            },
        })
    order = rng.permutation(n_samples)
    n_train = int(n_samples * train_fraction)
    n_val = int(n_samples * val_fraction)
    if min(n_train, n_val, n_samples - n_train - n_val) == 0:
        raise ValueError("each split must contain samples")
    split_indices = {
        "train": order[:n_train], "val": order[n_train:n_train+n_val],
        "test": order[n_train+n_val:]
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = ("X_measured_tavg_n1_d1", "X_clean_tavg_n0_d1",
            "Y_phys_tavg_n0_d1", "Y_bio_tavg_n0_d0",
            "expected_counts", "sampled_counts",
            "fdg_idif_dc_bqml", "pbr28_idif_dc_bqml")
    for split, indices in split_indices.items():
        data = {key: np.stack([samples[int(i)][key] for i in indices]).astype(
            np.int64 if key == "sampled_counts" else np.float32
        ) for key in keys}
        np.testing.assert_allclose(
            data["X_clean_tavg_n0_d1"][:, 0],
            data["Y_phys_tavg_n0_d1"].sum(axis=1), rtol=2e-6, atol=1e-3
        )
        np.savez_compressed(output_dir / f"{split}.npz", **data)
        (output_dir / f"{split}_metadata.json").write_text(json.dumps(
            [samples[int(i)]["metadata"] for i in indices], indent=2
        ) + "\n")
    (output_dir / "dataset_metadata.json").write_text(json.dumps({
        "units": {"TAC": "Bq/mL", "time": "minutes", "counts": "counts/frame"},
        "frame_edges_min": edges.tolist(),
        "frame_midpoints_min": ((edges[1:] + edges[:-1]) / 2).tolist(),
        "frame_durations_min": np.diff(edges).tolist(),
        "reference_doses_mci": {"FDG": fdg.reference_dose_mci, "PBR28": pbr.reference_dose_mci},
        "reference_doses_mbq": {"FDG": fdg.reference_dose_mci * MCI_TO_MBQ,
                                "PBR28": pbr.reference_dose_mci * MCI_TO_MBQ},
        "decay_reference": "beginning of each tracer's reference scan (assumed injection coincident)",
        "measured_input": "decay-corrected aortic whole-blood IDIF; PBR28 NOT parent-plasma corrected",
        "measurement_domain": "sum of physically decaying tracer contributions, then Poisson noise",
        "signal_source": signal_source,
        "fdg_kinetics": fdg_kinetics,
        "fdg_vb": fdg_vb,
        "fdg_delay_sec": fdg_delay_sec,
        "effective_sensitivity_cps_per_bq": effective_sensitivity_cps_per_bq,
        "effective_volume_ml": effective_volume_ml,
        "noise_warning": "idealized Poisson coefficient, NOT scanner-calibrated without independent validation",
        "fdg_offsets_min": list(fdg_offsets_min),
        "dose_relative_range": list(dose_relative_range),
        "seed": seed, "sample_count": n_samples,
        "channel_order": ["FDG", "PBR28"],
        "source_files_committed": False,
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fdg-csv", type=Path, required=True)
    parser.add_argument("--pbr-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/bqml_v1"))
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--pbr-duration-min", type=float, default=65.0)
    parser.add_argument("--fdg-offsets-min", type=float, nargs="+", default=[0, 5, 10])
    parser.add_argument("--effective-sensitivity-cps-per-bq", type=float, default=0.001)
    parser.add_argument("--effective-volume-ml", type=float, default=0.064)
    parser.add_argument("--signal-source", choices=("blood", "tissue-approx"),
                        default="tissue-approx")
    parser.add_argument("--fdg-kinetics", type=float, nargs=4,
                        metavar=("K1", "k2", "k3", "k4"),
                        help="Override FDG tissue rate constants (per minute)")
    parser.add_argument("--fdg-vb", type=float, default=0.0,
                        help="FDG whole-blood volume fraction, default 0")
    parser.add_argument("--fdg-delay-sec", type=float, default=0.0,
                        help="Additional FDG input delay in seconds, default 0")
    args = parser.parse_args()
    generate_dataset(args.fdg_csv, args.pbr_csv, args.output_dir,
                     n_samples=args.n_samples, seed=args.seed,
                     pbr_duration_min=args.pbr_duration_min,
                     fdg_offsets_min=tuple(args.fdg_offsets_min),
                     effective_sensitivity_cps_per_bq=args.effective_sensitivity_cps_per_bq,
                     effective_volume_ml=args.effective_volume_ml,
                     signal_source=args.signal_source,
                     fdg_kinetics=(dict(zip(("K1", "k2", "k3", "k4"), args.fdg_kinetics))
                                   if args.fdg_kinetics is not None else None),
                     fdg_vb=args.fdg_vb, fdg_delay_sec=args.fdg_delay_sec)
    print(f"Wrote Bq/mL TCN-compatible splits to {args.output_dir}")


if __name__ == "__main__":
    main()
