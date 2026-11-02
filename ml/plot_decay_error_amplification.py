"""Plot how radioactive decay correction amplifies TCN errors over time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRACERS = ("FDG", "PBR28")
COLORS = ("C0", "C1")


def rmse(error: np.ndarray, axis=None) -> np.ndarray:
    return np.sqrt(np.mean(error**2, axis=axis))


def scan_time_regions(frame_midpoints: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Split the scan into equal thirds of elapsed scan time."""
    start = 0.0
    stop = float(frame_midpoints[-1] + (frame_midpoints[-1] - frame_midpoints[-2]) / 2)
    boundaries = np.linspace(start, stop, 4)
    regions = []
    for index, name in enumerate(("Early", "Middle", "Late")):
        if index < 2:
            mask = (frame_midpoints >= boundaries[index]) & (frame_midpoints < boundaries[index + 1])
        else:
            mask = (frame_midpoints >= boundaries[index]) & (frame_midpoints <= boundaries[index + 1])
        regions.append((f"{name}\n{boundaries[index]:.1f}–{boundaries[index + 1]:.1f} min", mask))
    return regions


def choose_representative_sample(
    physical_error: np.ndarray,
    biological_error: np.ndarray,
    late_mask: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Choose the sample nearest the median late-error amplification ratio."""
    physical_late = rmse(physical_error[:, :, late_mask], axis=(1, 2))
    biological_late = rmse(biological_error[:, :, late_mask], axis=(1, 2))
    ratios = biological_late / np.maximum(physical_late, np.finfo(float).eps)
    index = int(np.argmin(np.abs(ratios - np.median(ratios))))
    return index, ratios


def plot_representative_sample(
    *,
    output_path: Path,
    index: int,
    metadata: list[dict],
    time: np.ndarray,
    physical_prediction: np.ndarray,
    physical_target: np.ndarray,
    biological_prediction: np.ndarray,
    biological_target: np.ndarray,
    decay_factors: np.ndarray,
    late_start: float,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    for channel, (tracer, color) in enumerate(zip(TRACERS, COLORS)):
        axes[0].plot(time, physical_target[index, channel], color=color, lw=2, label=f"{tracer} true")
        axes[0].plot(time, physical_prediction[index, channel], color=color, ls="--", lw=1.7, label=f"{tracer} predicted")
        axes[1].plot(time, biological_target[index, channel], color=color, lw=2, label=f"{tracer} true")
        axes[1].plot(time, biological_prediction[index, channel], color=color, ls="--", lw=1.7, label=f"{tracer} predicted")

        physical_abs_error = np.abs(physical_prediction[index, channel] - physical_target[index, channel])
        biological_abs_error = np.abs(biological_prediction[index, channel] - biological_target[index, channel])
        axes[2].plot(time, physical_abs_error, color=color, ls=":", lw=2, label=f"{tracer} physical error")
        axes[2].plot(time, biological_abs_error, color=color, lw=2, label=f"{tracer} biological error")

    for ax in axes:
        ax.axvspan(late_start, time[-1] + (time[-1] - time[-2]) / 2, color="gold", alpha=0.13, label="Late scan" if ax is axes[2] else None)
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=9)

    axes[0].set_title("1. Original TCN output: physical (decayed) TACs")
    axes[0].set_ylabel("Physical activity\n(kBq/mL)")
    axes[1].set_title("2. Same prediction after decay correction")
    axes[1].set_ylabel("Biological activity\n(kBq/mL)")
    axes[2].set_title("3. Absolute prediction error: correction enlarges late errors")
    axes[2].set_ylabel("Absolute error\n(kBq/mL)")
    axes[2].set_xlabel("Scan time (min)")

    late_mask = time >= late_start
    physical_late_rmse = float(rmse((physical_prediction[index] - physical_target[index])[:, late_mask]))
    biological_late_rmse = float(rmse((biological_prediction[index] - biological_target[index])[:, late_mask]))
    axes[2].text(
        0.98,
        0.95,
        f"Late RMSE: physical {physical_late_rmse:.3f} → biological {biological_late_rmse:.3f} kBq/mL\n"
        f"Observed amplification: {biological_late_rmse / physical_late_rmse:.2f}×",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )

    item = metadata[index]
    inverse_survival_end = 1.0 / decay_factors[index, :, -1]
    fig.suptitle(
        f"Decay correction amplifies tail error — representative test sample {item['sample_id']}\n"
        f"FDG delay {item['delta_min']:g} min | end-of-scan correction: "
        f"FDG {inverse_survival_end[0]:.2f}×, PBR28 {inverse_survival_end[1]:.2f}×",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_region_summary(
    *,
    output_path: Path,
    regions: list[tuple[str, np.ndarray]],
    physical_error: np.ndarray,
    biological_error: np.ndarray,
) -> dict:
    physical_by_region = np.asarray([rmse(physical_error[:, :, mask]) for _, mask in regions])
    biological_by_region = np.asarray([rmse(biological_error[:, :, mask]) for _, mask in regions])
    physical_by_tracer = np.asarray([rmse(physical_error[:, :, mask], axis=(0, 2)) for _, mask in regions])
    biological_by_tracer = np.asarray([rmse(biological_error[:, :, mask], axis=(0, 2)) for _, mask in regions])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(3)
    width = 0.36
    labels = [name for name, _ in regions]
    axes[0].bar(x - width / 2, physical_by_region, width, color="0.55", label="Physical RMSE")
    axes[0].bar(x + width / 2, biological_by_region, width, color="C3", label="Biological RMSE")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("RMSE (kBq/mL)")
    axes[0].set_title("Error before vs. after decay correction")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    for channel, (tracer, color) in enumerate(zip(TRACERS, COLORS)):
        ratio = biological_by_tracer[:, channel] / physical_by_tracer[:, channel]
        axes[1].plot(x, ratio, marker="o", lw=2.2, color=color, label=tracer)
        for xpos, value in zip(x, ratio):
            axes[1].annotate(f"{value:.2f}×", (xpos, value), xytext=(0, 7), textcoords="offset points", ha="center")
    axes[1].axhline(1.0, color="0.35", ls="--", lw=1, label="No amplification")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Biological RMSE / physical RMSE")
    axes[1].set_title("Error amplification grows toward the tail")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.suptitle("Decay-correction error amplification across all 100 test samples", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return {
        "region_definition": "Equal thirds of elapsed scan time; RMSE pools all samples, tracers, and frames in each region.",
        "regions": {
            labels[i].replace("\n", " "): {
                "physical_rmse_kBq_per_mL": float(physical_by_region[i]),
                "biological_rmse_kBq_per_mL": float(biological_by_region[i]),
                "amplification_ratio": float(biological_by_region[i] / physical_by_region[i]),
                "fdg_amplification_ratio": float(biological_by_tracer[i, 0] / physical_by_tracer[i, 0]),
                "pbr28_amplification_ratio": float(biological_by_tracer[i, 1] / physical_by_tracer[i, 1]),
            }
            for i in range(3)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/dual_tracer"))
    parser.add_argument("--physical-file", type=Path, default=Path("ml/experiments/mixture_30ep/evaluation/test_predictions.npz"))
    parser.add_argument("--biological-file", type=Path, default=Path("ml/experiments/mixture_30ep/decay_correction/biological_predictions.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("ml/experiments/mixture_30ep/decay_correction/error_amplification"))
    args = parser.parse_args()

    global_metadata = json.loads((args.data_dir / "dataset_metadata.json").read_text())
    sample_metadata = json.loads((args.data_dir / "test_metadata.json").read_text())
    time = np.asarray(global_metadata["frame_midpoints_min"], dtype=float)
    with np.load(args.physical_file) as data:
        physical_prediction = data["prediction_phys_tavg_n0_d1"].astype(float)
        physical_target = data["target_phys_tavg_n0_d1"].astype(float)
    with np.load(args.biological_file) as data:
        biological_prediction = data["prediction_bio_tavg_n0_d0"].astype(float)
        biological_target = data["target_bio_tavg_n0_d0"].astype(float)
        decay_factors = data["decay_factors_tavg"].astype(float)

    physical_error = physical_prediction - physical_target
    biological_error = biological_prediction - biological_target
    regions = scan_time_regions(time)
    representative, late_ratios = choose_representative_sample(physical_error, biological_error, regions[-1][1])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_representative_sample(
        output_path=args.output_dir / "representative_tail_amplification.png",
        index=representative,
        metadata=sample_metadata,
        time=time,
        physical_prediction=physical_prediction,
        physical_target=physical_target,
        biological_prediction=biological_prediction,
        biological_target=biological_target,
        decay_factors=decay_factors,
        late_start=float(time[regions[-1][1]][0]),
    )
    report = plot_region_summary(
        output_path=args.output_dir / "early_middle_late_rmse.png",
        regions=regions,
        physical_error=physical_error,
        biological_error=biological_error,
    )
    report["representative_test_index"] = representative
    report["representative_sample_id"] = int(sample_metadata[representative]["sample_id"])
    report["representative_late_amplification_ratio"] = float(late_ratios[representative])
    (args.output_dir / "error_amplification_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"representative_test_index={representative}")
    print(f"representative_sample_id={sample_metadata[representative]['sample_id']}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
