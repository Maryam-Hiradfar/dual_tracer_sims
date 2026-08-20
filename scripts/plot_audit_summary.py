"""Render each section of audit_report.json as a separate readable image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#365f91"
GREEN = "#3c763d"
ROW_ALT = "#f3f6f8"
BORDER = "#d0d4d8"


def _style_table(table, *, header_color: str = BLUE, font_size: int = 12) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.55)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(BORDER)
        cell.set_linewidth(0.7)
        cell.PAD = 0.08
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor(ROW_ALT)


def _save_table(
    rows,
    columns,
    title: str,
    output_path: Path,
    *,
    col_widths=None,
    header_color: str = BLUE,
    font_size: int = 12,
    figure_size=(11, 6),
):
    fig, ax = plt.subplots(figsize=figure_size)
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=col_widths,
    )
    _style_table(table, header_color=header_color, font_size=font_size)
    ax.set_title(title, fontsize=18, weight="bold", pad=20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_audit_sections(report_path: Path, output_dir: Path) -> list[Path]:
    report = json.loads(report_path.read_text())
    params = report["actual_parameter_statistics"]
    config = report["configuration"]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    overview_rows = [
        ["Audit status", report["status"]],
        ["Total samples", f"{report['actual_sample_count']:,}"],
        ["Train / validation / test", "800 / 100 / 100"],
        ["Tracer order", "PBR28 first → FDG second"],
        ["Scale range", "0.85–1.15 for both tracers"],
        ["Delay choices", "0, 5, 10 min"],
        ["Scanner / voxel", f"{config['scanner_name']} / {config['voxel_size_mm']:g} mm"],
        ["Noise model", config["noise_model"]],
        ["Global SNR", f"{report['noise_statistics']['global_snr_db']:.2f} dB"],
    ]
    paths.append(_save_table(
        overview_rows,
        ["Dataset property", "Value"],
        "Dataset configuration and overview",
        output_dir / "audit_overview.png",
        col_widths=(0.42, 0.58),
        figure_size=(10, 4.8),
    ))

    check_rows = [
        [name.replace("_", " ").capitalize(), "PASS" if value else "FAIL"]
        for name, value in report["checks"].items()
    ]
    check_path = output_dir / "audit_quality_checks.png"
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")
    table = ax.table(
        cellText=check_rows,
        colLabels=["Validation check", "Result"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=(0.78, 0.22),
    )
    _style_table(table, header_color=GREEN, font_size=12)
    for row, values in enumerate(check_rows, start=1):
        table[(row, 1)].set_text_props(
            color="#24733b" if values[1] == "PASS" else "#a22b2b",
            weight="bold",
        )
    ax.set_title("Automated dataset quality checks", fontsize=18, weight="bold", pad=20)
    fig.savefig(check_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(check_path)

    scale_rows = []
    for name in ("FDG_scale", "PBR28_scale"):
        stats = params[name]
        scale_rows.append([
            name.replace("_scale", ""),
            f"{stats['min']:.4f}",
            f"{stats['max']:.4f}",
            f"{stats['mean']:.4f}",
            f"{stats['std']:.4f}",
        ])
    paths.append(_save_table(
        scale_rows,
        ["Tracer", "Minimum", "Maximum", "Mean", "Std. dev."],
        "Actual sampled scale statistics",
        output_dir / "audit_scale_statistics.png",
        col_widths=(0.2, 0.2, 0.2, 0.2, 0.2),
        figure_size=(10, 2.5),
    ))

    delay_keys = ["0.0", "5.0", "10.0"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(delay_keys))
    for split, color in zip(
        ("train", "val", "test"),
        ("#4c78a8", "#f58518", "#54a24b"),
    ):
        values = np.array([
            params["delay_counts_by_split"][split][delay] for delay in delay_keys
        ])
        ax.bar(delay_keys, values, bottom=bottom, label=split.title(), color=color)
        for x, value, base in zip(range(len(values)), values, bottom):
            ax.text(x, base + value / 2, str(value), ha="center", va="center", color="white", fontsize=11)
        bottom += values
    for x, total in enumerate(bottom):
        ax.text(x, total + 7, f"Total: {int(total)}", ha="center", weight="bold")
    ax.set_title("Injection-delay distribution by split", fontsize=18, weight="bold", pad=16)
    ax.set_xlabel("FDG delay (min)", fontsize=12)
    ax.set_ylabel("Number of samples", fontsize=12)
    ax.set_ylim(0, max(bottom) * 1.13)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.2)
    delay_path = output_dir / "audit_delay_distribution.png"
    fig.tight_layout()
    fig.savefig(delay_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(delay_path)

    display_names = {
        "X_measured_tavg_n1_d1": "Noisy mixed input",
        "X_clean_tavg_n0_d1": "Clean mixed input",
        "Y_phys_tavg_n0_d1": "Physical FDG/PBR28 targets",
        "Y_bio_tavg_n0_d0": "Biological FDG/PBR28 targets",
        "expected_counts": "Expected scanner counts",
    }
    array_rows = []
    for name, stats in report["array_statistics"].items():
        percentiles = stats["percentiles"]
        array_rows.append([
            display_names[name],
            str(stats["shape"]),
            stats["dtype"],
            f"{stats['min']:.4g}",
            f"{percentiles['50']:.4g}",
            f"{percentiles['95']:.4g}",
            f"{stats['max']:.4g}",
            f"{stats['mean']:.4g}",
            f"{stats['std']:.4g}",
        ])
    paths.append(_save_table(
        array_rows,
        ["Array", "Shape", "Type", "Min", "Median", "95th %", "Max", "Mean", "Std. dev."],
        "Saved array statistics across all splits",
        output_dir / "audit_array_statistics.png",
        col_widths=(0.24, 0.14, 0.09, 0.08, 0.09, 0.09, 0.08, 0.08, 0.09),
        font_size=10,
        figure_size=(15, 3.5),
    ))

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("datasets/dual_tracer/audit_report.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/dual_tracer/audit_plots"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in plot_audit_sections(args.report, args.output_dir):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
