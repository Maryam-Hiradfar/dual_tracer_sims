"""Visualize all test predictions and analyze biological MSE by PET frame block."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


TRACERS = ("FDG", "PBR28")


def schedule_blocks(frame_durations: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Return contiguous blocks of equal-duration frames from frame_schedules.py."""
    boundaries = np.flatnonzero(~np.isclose(frame_durations[1:], frame_durations[:-1])) + 1
    groups = np.split(np.arange(len(frame_durations)), boundaries)
    blocks = []
    for indices in groups:
        seconds = frame_durations[indices[0]] * 60.0
        blocks.append((f"{len(indices)} × {seconds:g} s", indices))
    return blocks


def sample_mse(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    squared_error = (prediction - target) ** 2
    combined = squared_error.mean(axis=(1, 2))
    by_tracer = squared_error.mean(axis=2)
    return combined, by_tracer


def draw_sample(
    ax: plt.Axes,
    *,
    index: int,
    time: np.ndarray,
    mixture: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    metadata: list[dict],
    combined_mse: np.ndarray,
    tracer_mse: np.ndarray,
) -> None:
    item = metadata[index]
    ax.scatter(time, mixture[index, 0], s=10, alpha=0.45, color="0.45", label="Noisy physical mixture input")
    ax.plot(time, target[index, 0], color="C0", lw=1.8, label="FDG true biological")
    ax.plot(time, prediction[index, 0], color="C0", ls="--", lw=1.4, label="FDG predicted biological")
    ax.plot(time, target[index, 1], color="C1", lw=1.8, label="PBR28 true biological")
    ax.plot(time, prediction[index, 1], color="C1", ls="--", lw=1.4, label="PBR28 predicted biological")
    ax.set_title(
        f"test index {index} | sample {item['sample_id']} | delay {item['delta_min']:g} min\n"
        f"biological MSE: all={combined_mse[index]:.5f}, "
        f"FDG={tracer_mse[index, 0]:.5f}, PBR28={tracer_mse[index, 1]:.5f}"
    )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Activity concentration (kBq/mL)")
    ax.grid(alpha=0.22)


def save_all_sample_plots(
    *,
    output_dir: Path,
    time: np.ndarray,
    mixture: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    metadata: list[dict],
    combined_mse: np.ndarray,
    tracer_mse: np.ndarray,
) -> None:
    sample_dir = output_dir / "all_100_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "all_100_samples.pdf"
    with PdfPages(pdf_path) as pdf:
        for index in range(len(metadata)):
            fig, ax = plt.subplots(figsize=(10, 5.5))
            draw_sample(
                ax,
                index=index,
                time=time,
                mixture=mixture,
                prediction=prediction,
                target=target,
                metadata=metadata,
                combined_mse=combined_mse,
                tracer_mse=tracer_mse,
            )
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.90))
            fig.savefig(sample_dir / f"test_{index:03d}_sample_{metadata[index]['sample_id']}.png", dpi=160)
            pdf.savefig(fig)
            plt.close(fig)


def save_best_worst(
    *,
    output_path: Path,
    indices: list[int],
    labels: list[str],
    time: np.ndarray,
    mixture: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    metadata: list[dict],
    combined_mse: np.ndarray,
    tracer_mse: np.ndarray,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    for ax, index, rank_label in zip(axes.flat, indices, labels):
        draw_sample(
            ax,
            index=index,
            time=time,
            mixture=mixture,
            prediction=prediction,
            target=target,
            metadata=metadata,
            combined_mse=combined_mse,
            tracer_mse=tracer_mse,
        )
        ax.set_title(f"{rank_label}\n{ax.get_title()}", fontsize=10)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Best and worst biological TCN separations", y=0.995, fontsize=16)
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=5,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_mse_plots(
    *,
    output_path: Path,
    frame_midpoints: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    blocks: list[tuple[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    squared_error = (prediction - target) ** 2
    frame_mse = squared_error.mean(axis=0)
    block_mse = np.asarray([
        squared_error[:, :, indices].mean(axis=(0, 2)) for _, indices in blocks
    ])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].plot(frame_midpoints, frame_mse[0], marker="o", ms=3, color="C0", label="FDG")
    axes[0].plot(frame_midpoints, frame_mse[1], marker="o", ms=3, color="C1", label="PBR28")
    axes[0].set(title="MSE at every PET frame", xlabel="Frame midpoint (min)", ylabel="MSE ((kBq/mL)²)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    x = np.arange(len(blocks))
    width = 0.36
    axes[1].bar(x - width / 2, block_mse[:, 0], width, color="C0", label="FDG")
    axes[1].bar(x + width / 2, block_mse[:, 1], width, color="C1", label="PBR28")
    axes[1].set_xticks(x, [name for name, _ in blocks])
    axes[1].set(title="MSE by frame-schedule block", xlabel="Frame block", ylabel="MSE ((kBq/mL)²)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return frame_mse, block_mse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/dual_tracer"))
    parser.add_argument(
        "--prediction-file",
        type=Path,
        default=Path("ml/experiments/mixture_30ep/evaluation/test_predictions.npz"),
    )
    parser.add_argument(
        "--biological-file",
        type=Path,
        default=Path("ml/experiments/mixture_30ep/decay_correction/biological_predictions.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/experiments/mixture_30ep/test_biological_analysis"),
    )
    args = parser.parse_args()

    global_metadata = json.loads((args.data_dir / "dataset_metadata.json").read_text())
    metadata = json.loads((args.data_dir / "test_metadata.json").read_text())
    with np.load(args.prediction_file) as data:
        mixture = data["measured_tavg_n1_d1"].astype(np.float64)
    with np.load(args.biological_file) as data:
        prediction = data["prediction_bio_tavg_n0_d0"].astype(np.float64)
        target = data["target_bio_tavg_n0_d0"].astype(np.float64)

    if len(metadata) != 100:
        raise ValueError(f"expected the 100-sample test set, found {len(metadata)}")
    time = np.asarray(global_metadata["frame_midpoints_min"], dtype=float)
    durations = np.asarray(global_metadata["frame_durations_min"], dtype=float)
    blocks = schedule_blocks(durations)
    combined_mse, tracer_mse = sample_mse(prediction, target)
    order = np.argsort(combined_mse)
    best = order[:3].tolist()
    worst = order[-3:][::-1].tolist()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_all_sample_plots(
        output_dir=args.output_dir,
        time=time,
        mixture=mixture,
        prediction=prediction,
        target=target,
        metadata=metadata,
        combined_mse=combined_mse,
        tracer_mse=tracer_mse,
    )
    save_best_worst(
        output_path=args.output_dir / "best_3_and_worst_3.png",
        indices=[best[0], worst[0], best[1], worst[1], best[2], worst[2]],
        labels=["Best #1", "Worst #1", "Best #2", "Worst #2", "Best #3", "Worst #3"],
        time=time,
        mixture=mixture,
        prediction=prediction,
        target=target,
        metadata=metadata,
        combined_mse=combined_mse,
        tracer_mse=tracer_mse,
    )
    frame_mse, block_mse = save_mse_plots(
        output_path=args.output_dir / "mse_by_frame_and_schedule_block.png",
        frame_midpoints=time,
        prediction=prediction,
        target=target,
        blocks=blocks,
    )

    with (args.output_dir / "sample_mse.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("test_index", "sample_id", "delay_min", "combined_mse", "fdg_mse", "pbr28_mse"))
        for index, item in enumerate(metadata):
            writer.writerow((index, item["sample_id"], item["delta_min"], combined_mse[index], *tracer_mse[index]))
    with (args.output_dir / "frame_mse.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("frame_index", "midpoint_min", "duration_sec", "fdg_mse", "pbr28_mse"))
        for index in range(len(time)):
            writer.writerow((index, time[index], durations[index] * 60.0, frame_mse[0, index], frame_mse[1, index]))

    def sample_record(index: int) -> dict:
        return {
            "test_index": int(index),
            "sample_id": int(metadata[index]["sample_id"]),
            "delay_min": float(metadata[index]["delta_min"]),
            "combined_mse": float(combined_mse[index]),
            "fdg_mse": float(tracer_mse[index, 0]),
            "pbr28_mse": float(tracer_mse[index, 1]),
        }

    report = {
        "note": "These are the 100 held-out test samples, not training samples.",
        "mse_units": "(kBq/mL)^2",
        "sample_mse_definition": "Mean squared biological error across both tracers and all 62 frames.",
        "minimum_sample_mse": sample_record(int(order[0])),
        "maximum_sample_mse": sample_record(int(order[-1])),
        "best_3": [sample_record(index) for index in best],
        "worst_3": [sample_record(index) for index in worst],
        "schedule_blocks": [
            {
                "name": name,
                "frame_indices": [int(indices[0]), int(indices[-1])],
                "fdg_mse": float(block_mse[i, 0]),
                "pbr28_mse": float(block_mse[i, 1]),
                "combined_mse": float(block_mse[i].mean()),
            }
            for i, (name, indices) in enumerate(blocks)
        ],
    }
    (args.output_dir / "mse_summary.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"minimum_mse={combined_mse[order[0]]:.8g} test_index={order[0]} sample_id={metadata[order[0]]['sample_id']}")
    print(f"maximum_mse={combined_mse[order[-1]]:.8g} test_index={order[-1]} sample_id={metadata[order[-1]]['sample_id']}")
    print(f"best_3_test_indices={best}")
    print(f"worst_3_test_indices={worst}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
