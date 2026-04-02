from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from core.timegrid import TimeGrid
from tracers.fdg import FDGTracer
from tracers.pbr28 import PBR28Tracer

from separation.registry import create, available
from sim.config import ExperimentConfig

# ============================================================
# Builders
# ============================================================

def build_tracers(cfg: ExperimentConfig) -> tuple[PBR28Tracer, FDGTracer]:
    pbr = PBR28Tracer(
        name="PBR28",
        half_life_min=20.4,
        scale=cfg.tracer.pbr_scale,
        pbr28_params=(cfg.tracer.pbr_overrides or None),
    )

    fdg = FDGTracer(
        name="FDG",
        half_life_min=109.8,
        scale=cfg.tracer.fdg_scale,
        feng_params=(cfg.tracer.fdg_overrides or None),
    )

    return pbr, fdg


def build_timegrid(cfg: ExperimentConfig) -> TimeGrid:
    return TimeGrid(
        frame_edges=cfg.frame_edges_min,
        internal_dt_min=cfg.internal_dt_min,
    )


def build_algorithm(cfg: ExperimentConfig):
    alg_name = cfg.algorithm_name
    if alg_name not in available():
        raise ValueError(
            f"Separation algorithm '{alg_name}' not found. "
            f"Available algorithms: {available()}"
        )

    return create(
        alg_name,
        config=cfg.algorithm_config,
    )


def build_run_params(
    cfg: ExperimentConfig,
    pbr: PBR28Tracer,
    fdg: FDGTracer,
    timegrid: TimeGrid,
) -> dict[str, Any]:
    return {
        "experiment_id": cfg.experiment_id,
        "run_description": cfg.description,
        "tracers": ["FDG", "PBR28"],
        "fdg": {
            "name": fdg.name,
            "half_life_min": fdg.half_life_min,
            "scale": fdg.scale,
            "feng_params": fdg.feng_params,
        },
        "pbr": {
            "name": pbr.name,
            "half_life_min": pbr.half_life_min,
            "scale": pbr.scale,
            "pbr28_params": pbr.pbr28_params,
        },
        "delays_min": cfg.delays_min,
        "frame_edges_min": timegrid.frame_edges,
        "gamma_basis": asdict(cfg.gamma_library),
        "algorithm": asdict(cfg.algorithm_config),
        "random_seed": cfg.random_seed,
    }


def build_key_params_str(cfg: ExperimentConfig) -> str:
    return (
        f"{cfg.algorithm_name}, "
        f"alpha_stage_1={cfg.algorithm_config.alpha_stage_1}, "
        f"alpha_stage_2={cfg.algorithm_config.alpha_stage_2}, "
        f"num_iters={cfg.algorithm_config.num_iters}, "
        f"t_cut={cfg.algorithm_config.t_cut_min}, "
        f"FDG scale={cfg.tracer.fdg_scale}, "
        f"PBR scale={cfg.tracer.pbr_scale}, "
        f"gamma1={cfg.gamma_library.tracer_1_basis.n_t0}x{cfg.gamma_library.tracer_1_basis.n_tau}, "
        f"gamma2={cfg.gamma_library.tracer_2_basis.n_t0}x{cfg.gamma_library.tracer_2_basis.n_tau}"
    )

