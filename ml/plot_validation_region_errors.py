"""Compare whole-curve, tail, and peak-region validation errors per tracer.

Run from the repository root:
    python -m ml.plot_validation_region_errors --device cpu

The frozen checkpoint is evaluated only on val.npz. Peak windows are selected
from each true tracer curve separately in each domain; they are not selected
from predictions. All RMSEs weight frames equally, as in prototype training.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from core.timegrid import TimeGrid
from ml.decay_correct_predictions import decay_correct_physical_predictions
from ml.tcn import Model
from ml.train_tcn_prototype import DualTracerDataset, choose_device


TRACERS = ("FDG", "PBR28")
REGIONS = ("Whole curve", "Tail", "Peak region")
COLORS = ("#397cad", "#d98636")


def region_errors(prediction, target, time, tail_start, peak_half_window=1):
    """Return per-sample RMSE (N, 2, 3) and masks (N, 2, 3, T)."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (prediction.shape != target.shape or target.ndim != 3
            or target.shape[1] != 2 or target.shape[2] != len(time)):
        raise ValueError("expected matching (N, 2, T) arrays and T frame times")
    if len(target) == 0 or not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(target)):
        raise ValueError("predictions and targets must be nonempty and finite")
    if peak_half_window < 0:
        raise ValueError("peak half-window must be nonnegative")
    tail_mask = np.asarray(time) >= tail_start
    if not tail_mask.any():
        raise ValueError("tail interval contains no frame midpoints")
    masks = np.zeros((*target.shape[:2], 3, target.shape[2]), dtype=bool)
    masks[:, :, 0] = True
    masks[:, :, 1] = tail_mask
    peak_indices = target.argmax(axis=-1)
    masks[:, :, 2] = (
        np.abs(np.arange(target.shape[-1])[None, None, :] - peak_indices[..., None])
        <= peak_half_window
    )
    squared_error = (prediction - target) ** 2
    mse = (squared_error[:, :, None, :] * masks).sum(-1) / masks.sum(-1)
    return np.sqrt(mse), masks


def draw_comparison(errors, pooled, output_path, tail_start, scan_end, peak_half_window):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey="row")
    rng = np.random.default_rng(48)
    for row, domain in enumerate(("Physical", "Biological")):
        for channel, (tracer, color) in enumerate(zip(TRACERS, COLORS)):
            ax = axes[row, channel]
            values = errors[domain][:, channel]
            for region in range(3):
                ax.scatter(region + rng.uniform(-.16, .16, len(values)), values[:, region],
                           color=color, alpha=.22, s=15, edgecolors="none", zorder=2)
            bp = ax.boxplot([values[:, r] for r in range(3)], positions=range(3),
                            widths=.42, patch_artist=True, showfliers=False,
                            medianprops={"color": "#263445", "linewidth": 1.8},
                            whiskerprops={"color": color}, capprops={"color": color})
            for patch in bp["boxes"]:
                patch.set(facecolor=color, alpha=.18, edgecolor=color)
            aggregate = pooled[domain][channel]
            ax.scatter(range(3), aggregate, marker="D", s=46, color="#263445", zorder=4)
            for region, value in enumerate(aggregate):
                ax.annotate(f"{value:.3f}", (region, value), xytext=(20, 6),
                            textcoords="offset points", fontsize=10, color="#263445",
                            bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": 1})
            ax.set_xticks(range(3), REGIONS)
            ax.set_title(f"{tracer} · {domain.lower()} TACs", fontsize=13, weight="bold")
            ax.set_ylim(0, float(errors[domain].max()) * 1.15 + 1e-6)
            ax.set_xlim(-.5, 2.65)
            ax.grid(axis="y", alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
            if channel == 0:
                ax.set_ylabel("RMSE (kBq/mL) · lower is better")
    fig.suptitle("Where does the TCN make errors on validation scans?", fontsize=17, weight="bold", y=.975)
    fig.text(.5, .933, f"{len(errors['Physical'])} validation scans · frozen checkpoint · each dot is one scan",
             ha="center", fontsize=11)
    fig.text(.5, .115,
             "Dots: per-scan RMSE. Boxes: middle 50%; line: median; whiskers: up to 1.5 × IQR.\n"
             "Diamonds and numbers: pooled RMSE across all selected sample–frame pairs.",
             ha="center", fontsize=10, linespacing=1.5)
    caption = (
        f"Whole curve: all frames. Tail: frame midpoints in the final third of scan time ({tail_start:.1f}–{scan_end:.1f} min).\n"
        f"Peak region: true peak frame ± {peak_half_window} frame(s), clipped at scan boundaries, for each scan/tracer/domain.\n"
        "Frames have equal weight; regions may overlap. Biological errors include decay correction and its frame-averaging approximation."
    )
    fig.text(.5, .035, caption, ha="center", fontsize=9.5, linespacing=1.5)
    fig.subplots_adjust(top=.87, bottom=.22, left=.09, right=.97, hspace=.32, wspace=.14)
    fig.savefig(output_path, dpi=200, facecolor="white")
    plt.close(fig)
    return caption


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("ml/experiments/mixture_30ep/best_tcn_prototype.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/dual_tracer"))
    parser.add_argument("--output-dir", type=Path, default=Path("ml/experiments/mixture_30ep/validation_region_errors"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--peak-half-window", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.peak_half_window < 0:
        parser.error("batch size must be positive and peak half-window nonnegative")
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint["channel_order"] != list(TRACERS):
        raise ValueError("checkpoint channel order must be FDG, PBR28")
    dataset = DualTracerDataset(args.data_dir / "val.npz", checkpoint["normalization_scale"])
    model = Model(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        prediction = np.concatenate([
            model(x.to(device)).cpu().numpy()
            for x, _, _ in DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        ]) * dataset.scale
    target = dataset.y.numpy() * dataset.scale
    global_metadata = json.loads((args.data_dir / "dataset_metadata.json").read_text())
    metadata = json.loads((args.data_dir / "val_metadata.json").read_text())
    if len(metadata) != len(dataset):
        raise ValueError("validation metadata length does not match validation arrays")
    timegrid = TimeGrid(np.asarray(global_metadata["frame_edges_min"]),
                        internal_dt_min=float(global_metadata["internal_dt_min"]))
    time = np.asarray(global_metadata["frame_midpoints_min"])
    edges = np.asarray(global_metadata["frame_edges_min"])
    tail_start = float(edges[0] + (edges[-1] - edges[0]) * 2 / 3)
    biological_prediction, _ = decay_correct_physical_predictions(prediction, metadata, timegrid)
    with np.load(args.data_dir / "val.npz") as data:
        biological_target = data["Y_bio_tavg_n0_d0"].astype(np.float64)
    domains = {"Physical": (prediction, target), "Biological": (biological_prediction, biological_target)}
    errors, pooled, summary, rows = {}, {}, {}, []
    for domain, (pred, truth) in domains.items():
        values, masks = region_errors(pred, truth, time, tail_start, args.peak_half_window)
        errors[domain] = values
        sq = (pred.astype(np.float64) - truth.astype(np.float64)) ** 2
        pooled[domain] = np.sqrt((sq[:, :, None, :] * masks).sum(axis=(0, 3)) / masks.sum(axis=(0, 3)))
        summary[domain] = {}
        for ch, tracer in enumerate(TRACERS):
            summary[domain][tracer] = {}
            for r, region in enumerate(REGIONS):
                summary[domain][tracer][region] = {
                    "pooled_rmse": float(pooled[domain][ch, r]),
                    "median_sample_rmse": float(np.median(values[:, ch, r])),
                    "mean_sample_rmse": float(values[:, ch, r].mean()),
                    "selected_sample_frame_pairs": int(masks[:, ch, r].sum()),
                }
                for n, item in enumerate(metadata):
                    rows.append({"sample_id": item["sample_id"], "delay_min": item["delta_min"],
                                 "domain": domain, "tracer": tracer, "region": region,
                                 "rmse_kBq_mL": float(values[n, ch, r]),
                                 "frame_count": int(masks[n, ch, r].sum())})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / "validation_region_errors.png"
    caption = draw_comparison(errors, pooled, plot_path, tail_start, float(edges[-1]), args.peak_half_window)
    report = {"checkpoint": str(args.checkpoint), "checkpoint_epoch": int(checkpoint["epoch"]),
              "split": "validation", "samples": len(dataset), "units": "kBq/mL",
              "weighting": "equal per frame", "tail_start_min": tail_start,
              "peak_half_window_frames": args.peak_half_window, "caption": caption,
              "metrics": summary}
    (args.output_dir / "validation_region_errors_report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output_dir / "validation_region_errors_per_sample.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
