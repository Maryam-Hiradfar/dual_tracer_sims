# sim/delay_scale_experiments.py
#this file contains functions to build base simulation objects for delay and scale experiments
#these objects include time grid, gamma library, and FDG tracer (scale=1) that are common to all experiments
from core.timegrid import TimeGrid
from kinetics.basis_gamma import build_gamma_library
from tracers.fdg import FDGTracer
from tracers.pbr28 import PBR28Tracer
from sim.config import SimulationConfig
from sim.sweep import sweep_delays
import numpy as np

def build_base_sim_objects(config: SimulationConfig):
    rng = np.random.default_rng(config.rng_seed)

    timegrid = TimeGrid(
        frame_edges=config.frame_edges,
        internal_dt_min=1/60.0,
    )

    t_bio = timegrid.frame_mids
    gamma_lib, Gamma = build_gamma_library(
        t_bio,
        n_t0=config.n_t0,
        n_tau=config.n_tau,
    )

    # debug print to prove it's changing
    print(
        f"[DEBUG] gamma_lib for n_t0={config.n_t0}, n_tau={config.n_tau} "
        f"has shape {gamma_lib.shape}"
    )

    fdg = FDGTracer(
        name="FDG",
        half_life_min=109.8,
        scale=config.fdg_scale,
        feng_params=None,
    )

    return timegrid, gamma_lib, Gamma, fdg, rng

def make_pbr_tracer(scale: float) -> PBR28Tracer:
    """
    Factory function to construct a PBR tracer with a given scale value.
    Keeps tracer construction logic consistent across experiments.
    """
    return PBR28Tracer(
        name="PBR28",
        half_life_min=20.4,
        scale=scale,
        pbr28_params=None,  # __post_init__ will fill default params
    )


def run_delay_sweep_for_scale(
    scale: float,
    config: SimulationConfig,
    timegrid,
    gamma_lib,
    Gamma,
    fdg,
    rng,
):
    """
    Run the existing delay sweep for a single PBR scale value.

    Returns
    -------
    nrmse_pbr : np.ndarray  shape (n_delays,)
    nrmse_fdg : np.ndarray  shape (n_delays,)
    """
    # build a PBR tracer at this scale
    pbr = make_pbr_tracer(scale)

    results = sweep_delays(
        delays=config.delays,
        timegrid=timegrid,
        rac=pbr,          # your "PBR" tracer
        fdg=fdg,          # FDG tracer with scale=1
        gamma_lib=gamma_lib,
        Gamma=Gamma,
        rng=rng,
        use_joint=False,  # same as in your main
    )

    nrmse_pbr = np.array([r["nrmse_pbr_bio"] for r in results])
    nrmse_fdg = np.array([r["nrmse_fdg_bio"] for r in results])

    return nrmse_pbr, nrmse_fdg

def sweep_scale_and_delay(config: SimulationConfig):
    """
    Loop over PBR scales and build 2D NRMSE matrices:
    rows = scales, columns = delays.
    """
    timegrid, gamma_lib, Gamma, fdg, rng = build_base_sim_objects(config)

    n_scales = len(config.scale_values)
    n_delays = len(config.delays)

    nrmse_pbr_matrix = np.zeros((n_scales, n_delays))
    nrmse_fdg_matrix = np.zeros((n_scales, n_delays))

    for i, scale in enumerate(config.scale_values):
        nrmse_pbr, nrmse_fdg = run_delay_sweep_for_scale(
            scale,
            config,
            timegrid,
            gamma_lib,
            Gamma,
            fdg,
            rng,
        )
        nrmse_pbr_matrix[i, :] = nrmse_pbr
        nrmse_fdg_matrix[i, :] = nrmse_fdg

    return {
        "nrmse_pbr": nrmse_pbr_matrix,
        "nrmse_fdg": nrmse_fdg_matrix,
        "delays": config.delays,
        "scales": config.scale_values,
    }
