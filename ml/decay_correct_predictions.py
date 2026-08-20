"""Decay-correct saved physical TCN predictions into biological tracer TACs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from core.timegrid import TimeGrid
from utils.frame_average import frame_average


def frame_decay_factors(
    timegrid: TimeGrid,
    *,
    decay_constant: float,
    injection_time_min: float,
) -> np.ndarray:
    """Average radioactive survival over each PET frame."""
    time_since_injection = np.maximum(
        timegrid.t_internal - injection_time_min,
        0.0,
    )
    decay_internal = np.exp(-decay_constant * time_since_injection)
    return frame_average(
        timegrid.t_internal,
        decay_internal,
        timegrid.frame_edges,
    )


def decay_correct_physical_predictions(
    prediction_phys: np.ndarray,
    metadata: list[dict],
    timegrid: TimeGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert canonical physical channels (FDG, PBR28) to biological TACs."""
    if prediction_phys.shape != (len(metadata), 2, timegrid.n_frames):
        raise ValueError(
            "expected physical predictions shaped "
            f"({len(metadata)}, 2, {timegrid.n_frames}), "
            f"received {prediction_phys.shape}"
        )

    prediction_bio = np.empty_like(prediction_phys, dtype=np.float64)
    decay_factors = np.empty_like(prediction_phys, dtype=np.float64)
    for index, item in enumerate(metadata):
        if item["tracer1_name"] != "PBR28" or item["tracer2_name"] != "FDG":
            raise ValueError(
                "decay correction currently expects PBR28 first and FDG second"
            )

        pbr28_decay = frame_decay_factors(
            timegrid,
            decay_constant=float(item["lam1"]),
            injection_time_min=0.0,
        )
        fdg_decay = frame_decay_factors(
            timegrid,
            decay_constant=float(item["lam2"]),
            injection_time_min=float(item["delta_min"]),
        )
        decay_factors[index, 0] = fdg_decay
        decay_factors[index, 1] = pbr28_decay
        prediction_bio[index] = prediction_phys[index] / decay_factors[index]

    return prediction_bio, decay_factors


def tracer_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    metrics = {}
    for channel, name in enumerate(("FDG", "PBR28")):
        error = prediction[:, channel] - target[:, channel]
        mse = float(np.mean(error**2))
        target_range = float(np.max(target[:, channel]) - np.min(target[:, channel]))
        metrics[name] = {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(np.abs(error))),
            "nrmse_by_global_range": float(np.sqrt(mse) / target_range),
        }
    return metrics


def select_examples(metadata: list[dict]) -> list[int]:
    selected = []
    for delay in (0.0, 5.0, 10.0):
        candidates = [
            index
            for index, item in enumerate(metadata)
            if float(item["delta_min"]) == delay
        ]
        selected.append(min(
            candidates,
            key=lambda index: abs(
                metadata[index]["tracer2_scale"]
                / metadata[index]["tracer1_scale"]
                - 1.0
            ),
        ))
    return selected


def plot_biological_predictions(
    *,
    prediction_bio: np.ndarray,
    target_bio: np.ndarray,
    metadata: list[dict],
    frame_midpoints: np.ndarray,
    output_path: Path,
) -> None:
    selected = select_examples(metadata)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    for ax, index in zip(axes, selected):
        item = metadata[index]
        fdg_rmse = float(np.sqrt(np.mean(
            (prediction_bio[index, 0] - target_bio[index, 0]) ** 2
        )))
        pbr28_rmse = float(np.sqrt(np.mean(
            (prediction_bio[index, 1] - target_bio[index, 1]) ** 2
        )))
        ax.plot(frame_midpoints, target_bio[index, 0], color="C0", lw=2, label="FDG true biological")
        ax.plot(frame_midpoints, prediction_bio[index, 0], color="C0", ls="--", lw=1.7, label="FDG predicted biological")
        ax.plot(frame_midpoints, target_bio[index, 1], color="C1", lw=2, label="PBR28 true biological")
        ax.plot(frame_midpoints, prediction_bio[index, 1], color="C1", ls="--", lw=1.7, label="PBR28 predicted biological")
        ax.set_title(
            f"delay={item['delta_min']:g} min | sample {item['sample_id']}\n"
            f"FDG scale={item['tracer2_scale']:.3f}, "
            f"PBR28 scale={item['tracer1_scale']:.3f}"
        )
        ax.set_xlabel("Time (min)")
        ax.grid(True, alpha=0.25)
        ax.text(
            0.98,
            0.04,
            f"FDG RMSE: {fdg_rmse:.3f} kBq/mL\n"
            f"PBR28 RMSE: {pbr28_rmse:.3f} kBq/mL",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )
    axes[0].set_ylabel("Biological activity concentration (kBq/mL)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Decay-corrected TCN predictions versus true biological TACs", y=0.98)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_biological_predictions_with_input(
    *,
    measured_mixture: np.ndarray,
    prediction_bio: np.ndarray,
    target_bio: np.ndarray,
    metadata: list[dict],
    frame_midpoints: np.ndarray,
    output_path: Path,
) -> None:
    """Plot the physical mixture input alongside biological targets and predictions."""
    selected = select_examples(metadata)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    for ax, index in zip(axes, selected):
        item = metadata[index]
        fdg_rmse = float(np.sqrt(np.mean(
            (prediction_bio[index, 0] - target_bio[index, 0]) ** 2
        )))
        pbr28_rmse = float(np.sqrt(np.mean(
            (prediction_bio[index, 1] - target_bio[index, 1]) ** 2
        )))

        ax.scatter(
            frame_midpoints,
            measured_mixture[index, 0],
            color="0.45",
            alpha=0.55,
            s=14,
            label="Noisy physical mixture input",
        )
        ax.plot(
            frame_midpoints,
            target_bio[index, 0],
            color="C0",
            lw=2,
            label="FDG true biological",
        )
        ax.plot(
            frame_midpoints,
            prediction_bio[index, 0],
            color="C0",
            ls="--",
            lw=1.7,
            label="FDG predicted biological",
        )
        ax.plot(
            frame_midpoints,
            target_bio[index, 1],
            color="C1",
            lw=2,
            label="PBR28 true biological",
        )
        ax.plot(
            frame_midpoints,
            prediction_bio[index, 1],
            color="C1",
            ls="--",
            lw=1.7,
            label="PBR28 predicted biological",
        )
        ax.set_title(
            f"delay={item['delta_min']:g} min | sample {item['sample_id']}\n"
            f"FDG scale={item['tracer2_scale']:.3f}, "
            f"PBR28 scale={item['tracer1_scale']:.3f}"
        )
        ax.set_xlabel("Time (min)")
        ax.grid(True, alpha=0.25)
        ax.text(
            0.98,
            0.04,
            f"Biological FDG RMSE: {fdg_rmse:.3f} kBq/mL\n"
            f"Biological PBR28 RMSE: {pbr28_rmse:.3f} kBq/mL",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )

    axes[0].set_ylabel("Activity concentration (kBq/mL)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "TCN physical mixture input and decay-corrected biological outputs",
        y=0.98,
    )
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=5)
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_decay_correction(
    *,
    prediction_file: Path,
    data_dir: Path,
    output_dir: Path,
) -> dict:
    global_metadata = json.loads((data_dir / "dataset_metadata.json").read_text())
    sample_metadata = json.loads((data_dir / "test_metadata.json").read_text())
    timegrid = TimeGrid(
        np.asarray(global_metadata["frame_edges_min"]),
        internal_dt_min=float(global_metadata["internal_dt_min"]),
    )

    with np.load(prediction_file) as data:
        prediction_phys = data["prediction_phys_tavg_n0_d1"].astype(np.float64)
        target_phys = data["target_phys_tavg_n0_d1"].astype(np.float64)
        measured_mixture = data["measured_tavg_n1_d1"].astype(np.float64)
    with np.load(data_dir / "test.npz") as data:
        target_bio = data["Y_bio_tavg_n0_d0"].astype(np.float64)

    prediction_bio, decay_factors = decay_correct_physical_predictions(
        prediction_phys,
        sample_metadata,
        timegrid,
    )
    # This measures the small frame-averaging approximation even with true
    # physical TACs, independently of TCN prediction error.
    corrected_true_phys = target_phys / decay_factors

    metrics = tracer_metrics(prediction_bio, target_bio)
    correction_floor = tracer_metrics(corrected_true_phys, target_bio)
    overall_mse = float(np.mean((prediction_bio - target_bio) ** 2))

    output_dir.mkdir(parents=True, exist_ok=True)
    corrected_path = output_dir / "biological_predictions.npz"
    np.savez_compressed(
        corrected_path,
        prediction_bio_tavg_n0_d0=prediction_bio.astype(np.float32),
        target_bio_tavg_n0_d0=target_bio.astype(np.float32),
        decay_factors_tavg=decay_factors.astype(np.float32),
    )
    plot_path = output_dir / "biological_prediction_examples.png"
    plot_biological_predictions(
        prediction_bio=prediction_bio,
        target_bio=target_bio,
        metadata=sample_metadata,
        frame_midpoints=np.asarray(global_metadata["frame_midpoints_min"]),
        output_path=plot_path,
    )
    input_output_plot_path = (
        output_dir / "physical_input_and_biological_predictions.png"
    )
    plot_biological_predictions_with_input(
        measured_mixture=measured_mixture,
        prediction_bio=prediction_bio,
        target_bio=target_bio,
        metadata=sample_metadata,
        frame_midpoints=np.asarray(global_metadata["frame_midpoints_min"]),
        output_path=input_output_plot_path,
    )

    report = {
        "source_physical_predictions": str(prediction_file),
        "test_samples": len(sample_metadata),
        "channel_order": ["FDG", "PBR28"],
        "method": (
            "Divide each physical frame TAC by its frame-integrated tracer-"
            "specific radioactive survival factor."
        ),
        "units": {
            "mse": "(kBq/mL)^2",
            "rmse": "kBq/mL",
            "mae": "kBq/mL",
        },
        "overall_mse": overall_mse,
        "by_tracer": metrics,
        "frame_averaging_correction_floor_by_tracer": correction_floor,
        "corrected_prediction_file": str(corrected_path),
        "plot_file": str(plot_path),
        "physical_input_biological_output_plot_file": str(input_output_plot_path),
    }
    report_path = output_dir / "biological_test_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-file",
        type=Path,
        default=Path(
            "ml/experiments/mixture_30ep/evaluation/test_predictions.npz"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/dual_tracer"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/experiments/mixture_30ep/decay_correction"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_decay_correction(
        prediction_file=args.prediction_file,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    print(f"test_samples={report['test_samples']}")
    print(f"biological_overall_mse={report['overall_mse']:.8g}")
    for tracer in ("FDG", "PBR28"):
        values = report["by_tracer"][tracer]
        print(
            f"{tracer.lower()}_mse={values['mse']:.8g} "
            f"rmse={values['rmse']:.8g} mae={values['mae']:.8g}"
        )
    print(f"report={args.output_dir / 'biological_test_report.json'}")
    print(f"plot={report['plot_file']}")


if __name__ == "__main__":
    main()
