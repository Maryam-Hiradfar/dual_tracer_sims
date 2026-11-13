"""Evaluate a saved TCN checkpoint without retraining or updating its weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.tcn import Model
from ml.train_tcn_prototype import DualTracerDataset, choose_device


def _channel_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, dict[str, float]]:
    names = ("FDG", "PBR28")
    metrics = {}
    for channel, name in enumerate(names):
        error = prediction[:, channel] - target[:, channel]
        mse = float(np.mean(error**2))
        metrics[name] = {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(np.abs(error))),
        }
    return metrics


def _select_representative_indices(metadata: list[dict]) -> list[int]:
    """Select one test example for each delay, near a balanced scale ratio."""
    selected = []
    for delay in (0.0, 5.0, 10.0):
        candidates = [
            index
            for index, item in enumerate(metadata)
            if float(item["delta_min"]) == delay
        ]
        if not candidates:
            raise ValueError(f"no test sample found for delay {delay}")
        selected.append(min(
            candidates,
            key=lambda index: abs(
                metadata[index]["tracer2_scale"]
                / metadata[index]["tracer1_scale"]
                - 1.0
            ),
        ))
    return selected


def _plot_predictions(
    *,
    frame_midpoints: np.ndarray,
    measured: np.ndarray,
    clean: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    metadata: list[dict],
    selected_indices: list[int],
    output_path: Path,
    activity_unit: str,
) -> None:
    fig, axes = plt.subplots(1, len(selected_indices), figsize=(18, 5), sharex=True)
    for ax, index in zip(np.atleast_1d(axes), selected_indices):
        item = metadata[index]
        fdg_rmse = float(np.sqrt(np.mean(
            (prediction[index, 0] - target[index, 0]) ** 2
        )))
        pbr28_rmse = float(np.sqrt(np.mean(
            (prediction[index, 1] - target[index, 1]) ** 2
        )))
        ax.scatter(
            frame_midpoints,
            measured[index, 0],
            color="0.55",
            alpha=0.5,
            s=13,
            label="Noisy mixture",
        )
        ax.plot(frame_midpoints, clean[index, 0], color="black", lw=1.5, label="Clean mixture")
        ax.plot(frame_midpoints, target[index, 0], color="C0", lw=2, label="FDG true")
        ax.plot(frame_midpoints, prediction[index, 0], color="C0", ls="--", lw=1.7, label="FDG predicted")
        ax.plot(frame_midpoints, target[index, 1], color="C1", lw=2, label="PBR28 true")
        ax.plot(frame_midpoints, prediction[index, 1], color="C1", ls="--", lw=1.7, label="PBR28 predicted")
        ax.set_title(
            f"delay={item['delta_min']:g} min | sample {item['sample_id']}\n"
            f"FDG scale={item['tracer2_scale']:.3f}, "
            f"PBR28 scale={item['tracer1_scale']:.3f}"
        )
        ax.set_xlabel("Time (min)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(f"Activity concentration ({activity_unit})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("TCN test predictions versus true physical tracer TACs", y=0.98)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate(
    *,
    checkpoint_path: Path,
    data_dir: Path,
    output_dir: Path,
    batch_size: int,
    requested_device: str,
) -> dict:
    device = choose_device(requested_device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    scale = float(checkpoint["normalization_scale"])
    dataset = DualTracerDataset(data_dir / "test.npz", scale)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = Model(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    with torch.no_grad():
        for x, _, _ in loader:
            predictions.append(model(x.to(device)).cpu().numpy())
    prediction_norm = np.concatenate(predictions, axis=0)

    target_norm = dataset.y.numpy()
    measured_norm = dataset.x.numpy()
    clean_norm = dataset.x_clean.numpy()
    prediction = prediction_norm * scale
    target = target_norm * scale
    measured = measured_norm * scale
    clean = clean_norm * scale

    normalized_channel_metrics = _channel_metrics(prediction_norm, target_norm)
    physical_channel_metrics = _channel_metrics(prediction, target)
    normalized_overall_mse = float(np.mean((prediction_norm - target_norm) ** 2))
    physical_overall_mse = float(np.mean((prediction - target) ** 2))
    normalized_mixture_mse = float(
        np.mean((prediction_norm.sum(axis=1, keepdims=True) - clean_norm) ** 2)
    )
    physical_mixture_mse = float(
        np.mean((prediction.sum(axis=1, keepdims=True) - clean) ** 2)
    )

    metadata = json.loads((data_dir / "test_metadata.json").read_text())
    global_metadata = json.loads((data_dir / "dataset_metadata.json").read_text())
    activity_unit = global_metadata.get("units", {}).get("TAC", "kBq/mL")
    if activity_unit not in ("Bq/mL", "kBq/mL"):
        raise ValueError(f"unsupported activity unit: {activity_unit}")
    selected_indices = _select_representative_indices(metadata)

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "test_predictions.npz"
    np.savez_compressed(
        prediction_path,
        prediction_phys_tavg_n0_d1=prediction.astype(np.float32),
        target_phys_tavg_n0_d1=target.astype(np.float32),
        measured_tavg_n1_d1=measured.astype(np.float32),
        clean_tavg_n0_d1=clean.astype(np.float32),
    )
    from ml.tcn_diagnostic_plots import render_tcn_diagnostics
    additional_figures = render_tcn_diagnostics(
        np.asarray(global_metadata["frame_midpoints_min"]), measured, clean,
        target, prediction, metadata, global_metadata, selected_indices,
        output_dir, training_history=checkpoint_path.parent / "history.json",
    )
    plot_path = output_dir / "test_prediction_examples.png"

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_loss": float(checkpoint["val_loss"]),
        "device": str(device),
        "test_samples": len(dataset),
        "normalization_scale": scale,
        "channel_order": checkpoint["channel_order"],
        "training_config": checkpoint["training_config"],
        "normalized_metrics": {
            "overall_mse": normalized_overall_mse,
            "mixture_mse": normalized_mixture_mse,
            "by_tracer": normalized_channel_metrics,
        },
        "physical_metrics": {
            "units": {
                "mse": f"({activity_unit})^2",
                "rmse": activity_unit,
                "mae": activity_unit,
            },
            "overall_mse": physical_overall_mse,
            "mixture_mse": physical_mixture_mse,
            "by_tracer": physical_channel_metrics,
        },
        "representative_test_samples": [
            {
                "test_array_index": index,
                "sample_id": int(metadata[index]["sample_id"]),
                "delay_min": float(metadata[index]["delta_min"]),
            }
            for index in selected_indices
        ],
        "prediction_file": str(prediction_path),
        "plot_file": str(plot_path),
        "diagnostic_figures": additional_figures,
        "signal_origin": ("aortic blood input functions" if global_metadata.get("selected_rois")==["aorta"] else "ROI TAC columns"),
    }
    report_path = output_dir / "test_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("ml/experiments/mixture_30ep/best_tcn_prototype.pt"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/dual_tracer"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/experiments/mixture_30ep/evaluation"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        requested_device=args.device,
    )
    normalized = report["normalized_metrics"]
    physical = report["physical_metrics"]
    print(f"device={report['device']}")
    print(f"test_samples={report['test_samples']}")
    print(f"normalized_overall_mse={normalized['overall_mse']:.8g}")
    print(f"normalized_fdg_mse={normalized['by_tracer']['FDG']['mse']:.8g}")
    print(f"normalized_pbr28_mse={normalized['by_tracer']['PBR28']['mse']:.8g}")
    print(f"normalized_mixture_mse={normalized['mixture_mse']:.8g}")
    print(f"physical_overall_mse={physical['overall_mse']:.8g} {physical['units']['mse']}")
    print(f"report={args.output_dir / 'test_report.json'}")
    print(f"plot={report['plot_file']}")


if __name__ == "__main__":
    main()
