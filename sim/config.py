# sim/config.py
from dataclasses import dataclass
import numpy as np

@dataclass
class SimulationConfig:
    delays: np.ndarray
    scale_values: np.ndarray   # PBR:FDG scale ratios or absolute scales
    frame_edges: np.ndarray
    rng_seed: int = 42
    fdg_scale: float = 1.0
    n_t0: int = 30             # default gamma params
    n_tau: int = 30
