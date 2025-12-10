# sim/config.py
from dataclasses import dataclass
import numpy as np

@dataclass
class SimulationConfig:
    delays: np.ndarray
    scale_values: np.ndarray   # PBR:FDG scale ratios or absolute scales
    frame_edges: np.ndarray
    rng_seed: int = 42
