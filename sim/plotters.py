from __future__ import annotations
import numpy as np
from typing import Any

import matplotlib.pyplot as plt

from utils.run_logger import  get_figure_path

def plot_nrmse_vs_delay(
    delays_min: np.ndarray,
    results: list[dict[str, Any]],
    run_info,
    timestamp: str,
    ylim: tuple[float, float] | None = None,
) -> str:
    nrmse_pbr = np.array([r["nrmse_est_1_bio"] for r in results])
    nrmse_fdg = np.array([r["nrmse_est_2_bio"] for r in results])

    plt.figure(figsize=(7, 5))
    plt.plot(delays_min, nrmse_pbr, "-o", label="PBR NRMSE (bio)")
    plt.plot(delays_min, nrmse_fdg, "-o", label="FDG NRMSE (bio)")
    plt.xlabel("FDG injection delay Δ (min)")
    plt.ylabel("NRMSE")
    plt.title("Unmixing error vs injection delay")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()

    outpath = get_figure_path(run_info, f"nrmse_vs_delay_{timestamp}.png")
    plt.savefig(outpath, dpi=300)
    plt.close()
    return outpath


def plot_bio_tac_grid(
    results: list[dict[str, Any]],
    run_info,
    timestamp: str,
    ncols: int,
    nrows_per_fig: int,
    ylim: tuple[float, float],
) -> list[str]:
    fig_paths: list[str] = []

    plots_per_fig = ncols * nrows_per_fig
    n_results = len(results)
    n_figs = int(np.ceil(n_results / plots_per_fig))

    for f in range(n_figs):
        start = f * plots_per_fig
        end = min(start + plots_per_fig, n_results)
        chunk = results[start:end]
        nrows = int(np.ceil(len(chunk) / ncols))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(18, 4 * nrows),
            sharex=True, sharey=True
        )

        if nrows == 1:
            axes = np.expand_dims(axes, axis=0)

        for idx, r in enumerate(chunk):
            i, j = divmod(idx, ncols)
            ax = axes[i, j]

            D = r["Delta"]
            sim = r["sim"]

            t = sim["t_frames"]
            ct1_true = sim["Ct1_bio"]
            ct1_est = sim["Ct1_est_bio"]
            ct2_true = sim["Ct2_bio"]
            ct2_est = sim["Ct2_est_bio"]

            ax.plot(t, ct1_true, lw=2, label="PBR true")
            ax.plot(t, ct1_est, "--", lw=1.2, label="PBR est")
            ax.plot(t, ct2_true, lw=2, label="FDG true")
            ax.plot(t, ct2_est, "--", lw=1.2, label="FDG est")
            ax.set_ylim(*ylim)
            ax.set_title(f"Δ = {D} min")
            ax.grid(True, alpha=0.2)

        total_slots = nrows * ncols
        for k in range(len(chunk), total_slots):
            i, j = divmod(k, ncols)
            axes[i, j].axis("off")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=12)

        fig.text(0.5, 0.04, "Time (min)", ha="center", fontsize=14)
        fig.text(0.04, 0.5, "Biological concentration", va="center", rotation="vertical", fontsize=14)

        outpath = get_figure_path(run_info, f"bio_tac_grid_part{f+1}_{timestamp}.png")
        plt.tight_layout(rect=[0.03, 0.05, 1, 0.95])
        plt.savefig(outpath, dpi=300)
        plt.close()
        fig_paths.append(outpath)

    return fig_paths


def plot_measured_tac_grid(
    results: list[dict[str, Any]],
    run_info,
    timestamp: str,
    ncols: int,
    nrows_per_fig: int,
    ylim: tuple[float, float],
) -> list[str]:
    fig_paths: list[str] = []

    plots_per_fig = ncols * nrows_per_fig
    n_results = len(results)
    n_figs = int(np.ceil(n_results / plots_per_fig))

    for f in range(n_figs):
        start = f * plots_per_fig
        end = min(start + plots_per_fig, n_results)
        chunk = results[start:end]
        nrows = int(np.ceil(len(chunk) / ncols))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(18, 4 * nrows),
            sharex=True, sharey=True
        )

        if nrows == 1:
            axes = np.expand_dims(axes, axis=0)

        for idx, r in enumerate(chunk):
            i, j = divmod(idx, ncols)
            ax = axes[i, j]

            D = r["Delta"]
            sim = r["sim"]

            t = sim["t_frames"]
            ct1_true_meas = sim["Ct1_phys"]
            ct1_est_meas = sim["ct_1_est_meas"]
            ct2_true_meas = sim["Ct2_phys"]
            ct2_est_meas = sim["ct_2_est_meas"]

            ax.plot(t, ct1_true_meas, lw=2, label="PBR true")
            ax.plot(t, ct1_est_meas, "--", lw=1.2, label="PBR est")
            ax.plot(t, ct2_true_meas, lw=2, label="FDG true")
            ax.plot(t, ct2_est_meas, "--", lw=1.2, label="FDG est")
            ax.plot(t, ct1_true_meas + ct2_true_meas, color="black", lw=1, alpha=0.7, label="FDG + PBR sum")
            ax.scatter(t, sim["y_meas"], color="grey", s=10, alpha=0.5, label="Measured")
            ax.set_ylim(*ylim)
            ax.set_title(f"Δ = {D} min")
            ax.grid(True, alpha=0.2)

        total_slots = nrows * ncols
        for k in range(len(chunk), total_slots):
            i, j = divmod(k, ncols)
            axes[i, j].axis("off")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=12)

        fig.text(0.5, 0.04, "Time (min)", ha="center", fontsize=14)
        fig.text(
            0.04, 0.5,
            "Measured concentration, no individual decay correction",
            va="center", rotation="vertical", fontsize=14
        )

        outpath = get_figure_path(run_info, f"measured_tac_grid_part{f+1}_{timestamp}.png")
        plt.tight_layout(rect=[0.03, 0.05, 1, 0.95])
        plt.savefig(outpath, dpi=300)
        plt.close()
        fig_paths.append(outpath)

    return fig_paths


def plot_fdg_aligned(
    results: list[dict[str, Any]],
    run_info,
    timestamp: str,
    estimated: bool,
) -> str:
    plt.figure(figsize=(8, 5))

    for r in results:
        D = r["Delta"]
        sim = r["sim"]

        t = sim["t_frames"]
        y = sim["Ct2_est_bio"] if estimated else sim["Ct2_bio"]

        tau = np.maximum(t - D, 0.0)
        mask = t >= D
        plt.plot(tau[mask], y[mask], "--" if estimated else "-", label=f"Δ={D} min")

    plt.xlabel("Time since FDG injection τ (min)")
    plt.ylabel("Estimated FDG biological TAC" if estimated else "FDG biological TAC")
    plt.title("Estimated FDG TACs aligned by biological time" if estimated else "FDG biological TACs aligned by biological time")
    plt.xlim(0, 60)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()

    suffix = "fdg_aligned_estimated" if estimated else "fdg_aligned_true"
    outpath = get_figure_path(run_info, f"{suffix}_{timestamp}.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    return outpath


def plot_gamma_library(
    gamma_lib: list[np.ndarray],
    title: str,
    filename_stub: str,
    run_info,
    timestamp: str,
) -> str:
    plt.figure(figsize=(8, 5))
    for g in gamma_lib:
        plt.plot(g, alpha=0.3)
    plt.xlabel("Frame index")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    outpath = get_figure_path(run_info, f"{filename_stub}_{timestamp}.png")
    plt.savefig(outpath, dpi=300)
    plt.close()
    return outpath