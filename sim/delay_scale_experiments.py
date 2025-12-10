# sim/delay_scale_experiments.py
#this file contains functions to build base simulation objects for delay and scale experiments
#these objects include time grid, gamma library, and FDG tracer (scale=1) that are common to all experiments
from core.timegrid import TimeGrid
from sim.gamma_library import build_gamma_library
from tracers.fdg import FDGTracer
from tracers.pbr28 import PBR28Tracer

def build_base_sim_objects(config: SimulationConfig):
    """
    Build time grid, gamma library, and FDG tracer (scale=1).
    These don't depend on PBR scale, only on time axis.
    """
    rng = np.random.default_rng(config.rng_seed)

    timegrid = TimeGrid(
        frame_edges=config.frame_edges,
        internal_dt_min=1/60.0,   # or whatever you're using
    )

    t_bio = timegrid.frame_mids
    gamma_lib, Gamma = build_gamma_library(t_bio)

    fdg = FDGTracer(
        name="FDG",
        half_life_min=109.8,
        scale=1.0,          # fixed FDG reference
        feng_params=None,   # let __post_init__ fill defaults
    )

    return timegrid, gamma_lib, Gamma, fdg, rng
