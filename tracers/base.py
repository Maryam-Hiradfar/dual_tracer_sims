# tracers/base.py
from dataclasses import dataclass
import numpy as np

@dataclass
class Tracer:
    name: str
    half_life_min: float

    def aif(self, t_abs: np.ndarray, delta_min: float = 0.0) -> np.ndarray: #t_abs is the the absolute time vector that is the same as timegrid.t_internal or timegrid.frame_mids (?)
        raise NotImplementedError

    def true_params(self) -> dict:
        raise NotImplementedError
