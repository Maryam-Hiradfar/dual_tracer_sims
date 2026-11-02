from __future__ import annotations
import numpy as np
from typing import Any
from pathlib import Path

import matplotlib.pyplot as plt

from utils.run_logger import  get_figure_path
from kinetics.helpers.basis_gamma import gamma_shape


def plot_representative_dataset_samples(
    samples: list[dict[str, Any]],
    frame_midpoints_min: np.ndarray,
    output_dir: str | Path,
) -> list[Path]:
    """Plot representative saved samples in physical and biological domains."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t = np.asarray(frame_midpoints_min)

    fig, axes = plt.subplots(1, len(samples), figsize=(6 * len(samples), 5))
    axes = np.atleast_1d(axes)
    for ax, sample in zip(axes, samples):
        ax.scatter(
            t,
            sample["measured"],
            s=14,
            color="0.45",
            alpha=0.7,
            label="Noisy mixture",
        )
        ax.plot(t, sample["clean"], color="black", lw=2, label="Clean mixture")
        ax.plot(t, sample["fdg_phys"], lw=2, label="FDG physical")
        ax.plot(t, sample["pbr_phys"], lw=2, label="PBR28 physical")
        ax.set_title(sample["title"])
        ax.set_xlabel("Time (min)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Activity concentration (kBq/mL)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Representative physical-domain dataset samples", y=0.98)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    physical_path = output_dir / "representative_physical_tacs.png"
    fig.savefig(physical_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(samples), figsize=(6 * len(samples), 5))
    axes = np.atleast_1d(axes)
    for ax, sample in zip(axes, samples):
        ax.scatter(
            t,
            sample["measured"],
            s=14,
            color="0.55",
            alpha=0.55,
            label="Noisy physical mixture",
        )
        ax.plot(t, sample["clean"], color="black", lw=1.5, label="Clean physical mixture")
        ax.plot(t, sample["fdg_bio"], lw=2, label="FDG biological")
        ax.plot(t, sample["pbr_bio"], lw=2, label="PBR28 biological")
        ax.set_title(sample["title"])
        ax.set_xlabel("Time (min)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Activity concentration (kBq/mL)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "Biological TACs with physical scanner mixtures (different domains)",
        y=0.98,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    biological_path = output_dir / "representative_biological_tacs.png"
    fig.savefig(biological_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return [physical_path, biological_path]

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

            sim = r["sim"]
            D = sim.Delta
            t = sim.t_int


            #t = sim.t_frames
            ct1_true = sim.Ct1_bio_tint_n0_d0
            ct1_est = sim.Ct1_est_bio_tint_n1_d0
            ct2_true = sim.Ct2_bio_tint_n0_d0
            ct2_est = sim.Ct2_est_bio_tint_n1_d0

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
            sim = r["sim"]
            D = sim.Delta 
            
          

            t = sim.t_int
            t_meas = sim.t_frames
            ct1_true_meas = sim.Ct1_phys_tint_n0_d1
            ct1_est_meas = sim.ct_1_est_meas_tint_n1_d1
            ct2_true_meas = sim.Ct2_phys_tint_n0_d1
            ct2_est_meas = sim.ct_2_est_meas_tint_n1_d1
    
            ax.plot(t, ct1_true_meas, lw=2, label="PBR true")
            ax.plot(t, ct1_est_meas, "--", lw=1.2, label="PBR est")
            ax.plot(t, ct2_true_meas, lw=2, label="FDG true")
            ax.plot(t, ct2_est_meas, "--", lw=1.2, label="FDG est")
            ax.plot(t, ct1_true_meas + ct2_true_meas, color="black", lw=1, alpha=0.7, label="FDG + PBR sum")
            ax.scatter(t_meas, sim.y_meas_tavg_n1_d1, color="grey", s=10, alpha=0.5, label="Measured")
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
        sim = r["sim"]
        D = sim.Delta
        

        t = sim.t_int
        y = sim.Ct2_est_bio_tint_n1_d0 if estimated else sim.Ct2_bio_tint_n0_d0

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
    t: np.ndarray,
    params: list[tuple[float, float]],
    scale: float,
    title: str,
    filename_stub: str,
    run_info,
    timestamp: str,
) -> str:
    t_plot = np.linspace(float(t[0]), float(t[-1]), 1000)

    plt.figure(figsize=(8, 5))
    for t0, tau in params:
        g_plot = gamma_shape(t_plot, t0, tau, scale=scale)
        plt.plot(t_plot, g_plot, alpha=0.3)

    plt.xlabel("Time (min)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    outpath = get_figure_path(run_info, f"{filename_stub}_{timestamp}.png")
    plt.savefig(outpath, dpi=300)
    plt.close()
    return outpath
