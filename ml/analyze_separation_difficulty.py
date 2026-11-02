"""Relate TCN separation error to simulation parameters and curve difficulty."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


TRACERS = ("FDG", "PBR28")


def duration_weighted_area(curve: np.ndarray, durations: np.ndarray) -> float:
    return float(np.sum(np.asarray(curve, dtype=float) * durations))


def difficulty_scores(
    fdg: np.ndarray,
    pbr28: np.ndarray,
    durations: np.ndarray,
) -> dict[str, float]:
    """Return interpretable 0--100 oracle difficulty descriptors."""
    fdg_area = duration_weighted_area(fdg, durations)
    pbr_area = duration_weighted_area(pbr28, durations)
    if fdg_area <= 0 or pbr_area <= 0:
        return {
            "temporal_overlap_percent": 0.0,
            "shared_activity_percent": 0.0,
            "amplitude_similarity_percent": 0.0,
            "entropy_ambiguity_percent": 0.0,
        }

    temporal_overlap = np.sum(
        durations * np.minimum(fdg / fdg_area, pbr28 / pbr_area)
    )
    shared_activity = (
        2.0 * np.sum(durations * np.minimum(fdg, pbr28))
        / (fdg_area + pbr_area)
    )
    amplitude_similarity = 2.0 * min(fdg_area, pbr_area) / (fdg_area + pbr_area)

    mixture = fdg + pbr28
    active = mixture > 0
    fdg_fraction = np.zeros_like(mixture, dtype=float)
    fdg_fraction[active] = fdg[active] / mixture[active]
    entropy = np.zeros_like(mixture, dtype=float)
    interior = active & (fdg_fraction > 0) & (fdg_fraction < 1)
    fraction = fdg_fraction[interior]
    entropy[interior] = -(
        fraction * np.log(fraction)
        + (1.0 - fraction) * np.log(1.0 - fraction)
    ) / np.log(2.0)
    activity_weights = durations * mixture
    entropy_ambiguity = float(
        np.sum(activity_weights * entropy) / np.sum(activity_weights)
    )

    return {
        "temporal_overlap_percent": 100.0 * float(temporal_overlap),
        "shared_activity_percent": 100.0 * float(shared_activity),
        "amplitude_similarity_percent": 100.0 * float(amplitude_similarity),
        "entropy_ambiguity_percent": 100.0 * entropy_ambiguity,
    }


def correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    finite = np.isfinite(x) & np.isfinite(y)
    x_valid, y_valid = x[finite], y[finite]
    if len(x_valid) < 3 or np.isclose(np.ptp(x_valid), 0):
        return {
            "n": int(len(x_valid)),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
        }
    pearson = pearsonr(x_valid, y_valid)
    spearman = spearmanr(x_valid, y_valid)
    return {
        "n": int(len(x_valid)),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def tracer_scale(item: dict, tracer: str) -> float:
    if item["tracer1_name"] == tracer:
        return float(item["tracer1_scale"])
    if item["tracer2_name"] == tracer:
        return float(item["tracer2_scale"])
    raise ValueError(f"sample {item['sample_id']} does not contain {tracer}")


def build_rows(
    *,
    metadata: list[dict],
    durations: np.ndarray,
    measured: np.ndarray,
    clean: np.ndarray,
    true_phys: np.ndarray,
    predicted_phys: np.ndarray,
    true_bio: np.ndarray,
    predicted_bio: np.ndarray,
    expected_counts: np.ndarray,
) -> list[dict[str, float | int | str]]:
    rows = []
    for index, item in enumerate(metadata):
        physical_error = predicted_phys[index] - true_phys[index]
        biological_error = predicted_bio[index] - true_bio[index]
        fdg_scale = tracer_scale(item, "FDG")
        pbr_scale = tracer_scale(item, "PBR28")
        scores = difficulty_scores(
            true_phys[index, 0],
            true_phys[index, 1],
            durations,
        )
        noise = measured[index, 0] - clean[index, 0]
        clean_rms = float(np.sqrt(np.mean(clean[index, 0] ** 2)))
        fdg_phys_area = duration_weighted_area(true_phys[index, 0], durations)
        pbr_phys_area = duration_weighted_area(true_phys[index, 1], durations)
        fdg_bio_area = duration_weighted_area(true_bio[index, 0], durations)
        pbr_bio_area = duration_weighted_area(true_bio[index, 1], durations)
        rows.append({
            "test_index": index,
            "sample_id": int(item["sample_id"]),
            "child_seed": int(item["seed"]),
            "split": "test",
            "fdg_delay_min": float(item["delta_min"]),
            "fdg_scale": fdg_scale,
            "pbr28_scale": pbr_scale,
            "fdg_to_pbr28_scale_ratio": fdg_scale / pbr_scale,
            "physical_overall_mse": float(np.mean(physical_error**2)),
            "physical_overall_rmse": float(np.sqrt(np.mean(physical_error**2))),
            "physical_fdg_rmse": float(np.sqrt(np.mean(physical_error[0] ** 2))),
            "physical_pbr28_rmse": float(np.sqrt(np.mean(physical_error[1] ** 2))),
            "biological_overall_rmse": float(np.sqrt(np.mean(biological_error**2))),
            "biological_fdg_rmse": float(np.sqrt(np.mean(biological_error[0] ** 2))),
            "biological_pbr28_rmse": float(np.sqrt(np.mean(biological_error[1] ** 2))),
            "decay_rmse_amplification": float(
                np.sqrt(np.mean(biological_error**2))
                / np.sqrt(np.mean(physical_error**2))
            ),
            "mixture_noise_rmse": float(np.sqrt(np.mean(noise**2))),
            "relative_mixture_noise": float(np.sqrt(np.mean(noise**2)) / clean_rms),
            "total_expected_counts": float(np.sum(expected_counts[index])),
            "median_expected_counts": float(np.median(expected_counts[index])),
            "fdg_decay_attenuation_percent": 100.0 * (1.0 - fdg_phys_area / fdg_bio_area),
            "pbr28_decay_attenuation_percent": 100.0 * (1.0 - pbr_phys_area / pbr_bio_area),
            **scores,
        })
    return rows


def values(rows: list[dict], name: str) -> np.ndarray:
    return np.asarray([row[name] for row in rows], dtype=float)


def scatter_with_trend(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    xlabel: str,
    title: str,
) -> None:
    ax.scatter(x, y, s=28, alpha=0.65, edgecolor="none")
    if not np.isclose(np.ptp(x), 0):
        order = np.argsort(x)
        slope, intercept = np.polyfit(x, y, 1)
        ax.plot(x[order], slope * x[order] + intercept, color="0.2", lw=1.5)
    stats = correlation(x, y)
    rho = stats["spearman_rho"]
    r = stats["pearson_r"]
    label = "constant feature" if rho is None else f"Pearson r={r:.2f} | Spearman ρ={rho:.2f}"
    ax.set(title=f"{title}\n{label}", xlabel=xlabel, ylabel="Physical overall RMSE (kBq/mL)")
    ax.grid(alpha=0.23)


def plot_parameters(rows: list[dict], output_path: Path) -> None:
    y = values(rows, "physical_overall_rmse")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    delays = values(rows, "fdg_delay_min")
    groups = [y[delays == delay] for delay in (0, 5, 10)]
    axes[0, 0].boxplot(groups, tick_labels=("0", "5", "10"), showmeans=True)
    rng = np.random.default_rng(0)
    for position, group in enumerate(groups, start=1):
        axes[0, 0].scatter(
            position + rng.uniform(-0.08, 0.08, len(group)),
            group,
            s=20,
            alpha=0.55,
        )
    axes[0, 0].set(
        title="FDG delay groups: each dot is one sample",
        xlabel="FDG delay (min)",
        ylabel="Physical overall RMSE (kBq/mL)",
    )
    axes[0, 0].grid(axis="y", alpha=0.23)

    specs = (
        ("fdg_scale", "FDG AIF scale", "FDG scale"),
        ("pbr28_scale", "PBR28 AIF scale", "PBR28 scale"),
        ("fdg_to_pbr28_scale_ratio", "FDG/PBR28 scale ratio", "Scale ratio"),
        ("mixture_noise_rmse", "Mixture noise RMSE (kBq/mL)", "Realized noise"),
    )
    for ax, (name, xlabel, title) in zip(axes.flat[1:], specs):
        scatter_with_trend(ax, values(rows, name), y, xlabel=xlabel, title=title)
    axes.flat[-1].axis("off")
    fig.suptitle("Simulation parameters and noise versus TCN physical error", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_difficulty(rows: list[dict], output_path: Path) -> None:
    y = values(rows, "physical_overall_rmse")
    specs = (
        ("temporal_overlap_percent", "Temporal overlap (%)", "Area-normalized temporal overlap"),
        ("shared_activity_percent", "Shared activity (%)", "Raw shared-activity overlap"),
        ("amplitude_similarity_percent", "Amplitude similarity (%)", "Integrated amplitude similarity"),
        ("entropy_ambiguity_percent", "Mixture ambiguity (%)", "Activity-weighted contribution entropy"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (name, xlabel, title) in zip(axes.flat, specs):
        scatter_with_trend(ax, values(rows, name), y, xlabel=xlabel, title=title)
    fig.suptitle("Oracle curve-difficulty descriptors versus TCN physical RMSE", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/dual_tracer"))
    parser.add_argument(
        "--physical-predictions",
        type=Path,
        default=Path("ml/experiments/mixture_30ep/evaluation/test_predictions.npz"),
    )
    parser.add_argument(
        "--biological-predictions",
        type=Path,
        default=Path("ml/experiments/mixture_30ep/decay_correction/biological_predictions.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/experiments/mixture_30ep/difficulty_analysis"),
    )
    args = parser.parse_args()

    global_metadata = json.loads((args.data_dir / "dataset_metadata.json").read_text())
    metadata = json.loads((args.data_dir / "test_metadata.json").read_text())
    durations = np.asarray(global_metadata["frame_durations_min"], dtype=float)
    with np.load(args.data_dir / "test.npz") as data:
        measured = data["X_measured_tavg_n1_d1"].astype(float)
        clean = data["X_clean_tavg_n0_d1"].astype(float)
        true_phys = data["Y_phys_tavg_n0_d1"].astype(float)
        true_bio = data["Y_bio_tavg_n0_d0"].astype(float)
        expected_counts = data["expected_counts"].astype(float)
    with np.load(args.physical_predictions) as data:
        predicted_phys = data["prediction_phys_tavg_n0_d1"].astype(float)
    with np.load(args.biological_predictions) as data:
        predicted_bio = data["prediction_bio_tavg_n0_d0"].astype(float)

    rows = build_rows(
        metadata=metadata,
        durations=durations,
        measured=measured,
        clean=clean,
        true_phys=true_phys,
        predicted_phys=predicted_phys,
        true_bio=true_bio,
        predicted_bio=predicted_bio,
        expected_counts=expected_counts,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_sample_difficulty.csv", rows)
    plot_parameters(rows, args.output_dir / "parameters_vs_physical_rmse.png")
    plot_difficulty(rows, args.output_dir / "difficulty_vs_physical_rmse.png")

    y = values(rows, "physical_overall_rmse")
    feature_names = (
        "fdg_delay_min",
        "fdg_scale",
        "pbr28_scale",
        "fdg_to_pbr28_scale_ratio",
        "mixture_noise_rmse",
        "relative_mixture_noise",
        "total_expected_counts",
        "median_expected_counts",
        "fdg_decay_attenuation_percent",
        "pbr28_decay_attenuation_percent",
        "temporal_overlap_percent",
        "shared_activity_percent",
        "amplitude_similarity_percent",
        "entropy_ambiguity_percent",
    )
    relationships = {
        name: correlation(values(rows, name), y) for name in feature_names
    }
    delay_summary = {
        str(delay): {
            "n": int(np.sum(values(rows, "fdg_delay_min") == delay)),
            "mean_rmse": float(np.mean(y[values(rows, "fdg_delay_min") == delay])),
            "median_rmse": float(np.median(y[values(rows, "fdg_delay_min") == delay])),
        }
        for delay in (0.0, 5.0, 10.0)
    }
    report = {
        "samples": len(rows),
        "primary_outcome": "per-sample physical RMSE pooled equally over 2 tracers and 62 frames",
        "rmse_units": "kBq/mL",
        "plot_semantics": {
            "scatter": "one dot is one complete held-out test sample; RMSE pools its 124 tracer-frame errors",
            "delay": "each dot is one sample; box shows the distribution within a delay group; line is median; triangle is mean",
            "trend_line": "ordinary univariate least-squares line for visualization only",
        },
        "difficulty_definitions": {
            "temporal_overlap_percent": "duration-weighted overlap after separately area-normalizing each true physical TAC",
            "shared_activity_percent": "2 times duration-weighted shared area divided by summed tracer areas; includes amplitude imbalance",
            "amplitude_similarity_percent": "2 times the smaller tracer area divided by summed tracer areas",
            "entropy_ambiguity_percent": "activity- and duration-weighted binary entropy of the FDG/PBR28 contribution fraction in each frame",
        },
        "decay_correction_note": (
            "Decay-correction amplification is saved per sample but is not treated as an "
            "independent predictor because it contains physical RMSE in its denominator."
        ),
        "physical_vs_biological_rmse": correlation(
            values(rows, "physical_overall_rmse"),
            values(rows, "biological_overall_rmse"),
        ),
        "delay_group_summary": delay_summary,
        "correlations_with_physical_overall_rmse": relationships,
    }
    (args.output_dir / "difficulty_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"samples={len(rows)}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
