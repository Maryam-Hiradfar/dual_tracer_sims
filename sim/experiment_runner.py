from  __future__ import annotations
import numpy as np
from datetime import datetime
from sim.plotters import (
    plot_nrmse_vs_delay,
    plot_bio_tac_grid,
    plot_measured_tac_grid,
    plot_fdg_aligned,
    plot_gamma_library,
)
from core.timegrid import TimeGrid
from tracers.fdg import FDGTracer
from tracers.pbr28 import PBR28Tracer

from kinetics.helpers.basis_gamma import build_uniform_gamma_library
from sim.sweep import sweep_delays

from sim.config import ExperimentConfig, ExperimentSummary
from sim.builders import build_tracers, build_timegrid, build_algorithm, build_run_params
from utils.run_logger import start_run, finalize_run, build_key_params_str




# ============================================================
# Core runner
# ============================================================

def run_single_experiment(cfg: ExperimentConfig) -> ExperimentSummary:
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")

    pbr, fdg = build_tracers(cfg)
    timegrid = build_timegrid(cfg)
    if cfg.gamma_library is not None:
        gamma_lib_1, Gamma_1 = build_uniform_gamma_library(
            timegrid.frame_mids,
            cfg.gamma_library.tracer_1_basis,
        )
        gamma_lib_2, Gamma_2 = build_uniform_gamma_library(
            timegrid.frame_mids,
            cfg.gamma_library.tracer_2_basis,
        )

    elif cfg.kinetics_library is not None:
        # build kinetics-informed libraries here
        raise NotImplementedError("Kinetics-informed library build not wired yet.")

    else:
        raise ValueError(
            "ExperimentConfig must provide either gamma_library or kinetics_library."
        )

    separation_alg = build_algorithm(cfg)

    run_info = start_run(
        build_run_params(cfg, pbr, fdg, timegrid),
        description=cfg.description,
        output_root=cfg.output_root,
    )

    rng = np.random.default_rng(cfg.random_seed)

    results = sweep_delays(
        delays=cfg.delays_min,
        timegrid=timegrid,
        rac=pbr,
        fdg=fdg,
        gamma_lib_1=gamma_lib_1,
        Gamma_1=Gamma_1,
        gamma_lib_2=gamma_lib_2,
        Gamma_2=Gamma_2,
        rng=rng,
        separation_alg=separation_alg,
        # if your refactor removed context, stop here.
        # if sweep_delays still temporarily expects t_cut/context-like info,
        # pass only the typed pieces it still needs.
    )


    fig_paths: list[str] = []

    if cfg.plots.save_nrmse_vs_delay:
        fig_paths.append(
            plot_nrmse_vs_delay(
                delays_min=cfg.delays_min,
                results=results,
                run_info=run_info,
                timestamp=timestamp,
                ylim=cfg.plots.nrmse_ylim,
            )
        )

    if cfg.plots.save_bio_tac_grid:
        fig_paths.extend(
            plot_bio_tac_grid(
                results=results,
                run_info=run_info,
                timestamp=timestamp,
                ncols=cfg.plots.tac_grid_ncols,
                nrows_per_fig=cfg.plots.tac_grid_nrows_per_fig,
                ylim=cfg.plots.tac_ylim,
            )
        )

    if cfg.plots.save_measured_tac_grid:
        fig_paths.extend(
            plot_measured_tac_grid(
                results=results,
                run_info=run_info,
                timestamp=timestamp,
                ncols=cfg.plots.tac_grid_ncols,
                nrows_per_fig=cfg.plots.tac_grid_nrows_per_fig,
                ylim=cfg.plots.tac_ylim,
            )
        )

    if cfg.plots.save_fdg_aligned_true:
        fig_paths.append(plot_fdg_aligned(results, run_info, timestamp, estimated=False))

    if cfg.plots.save_fdg_aligned_est:
        fig_paths.append(plot_fdg_aligned(results, run_info, timestamp, estimated=True))

    if cfg.plots.save_gamma_library:
        fig_paths.append(
            plot_gamma_library(
                gamma_lib_1,
                title="Gamma basis library for tracer 1",
                filename_stub="gamma_library_tracer_1",
                run_info=run_info,
                timestamp=timestamp,
            )
        )
        fig_paths.append(
            plot_gamma_library(
                gamma_lib_2,
                title="Gamma basis library for tracer 2",
                filename_stub="gamma_library_tracer_2",
                run_info=run_info,
                timestamp=timestamp,
            )
        )

    finalize_run(
        run_info,
        figure_filenames=fig_paths,
        key_params_str=build_key_params_str(cfg),
        notes=cfg.notes.strip(),
    )

    mean_nrmse_pbr = float(np.mean([r["nrmse_est_1_bio"] for r in results]))
    mean_nrmse_fdg = float(np.mean([r["nrmse_est_2_bio"] for r in results]))

    return ExperimentSummary(
        experiment_id=cfg.experiment_id,
        run_dir=str(run_info.figure_dir),
        n_conditions=len(results),
        mean_nrmse_pbr=mean_nrmse_pbr,
        mean_nrmse_fdg=mean_nrmse_fdg,
    )




