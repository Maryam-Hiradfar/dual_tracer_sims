# tracers/pbr28.py
from dataclasses import dataclass
import numpy as np
from .base import Tracer

def pbr28_aif(t, scale = 1.0):
    A1, A2, A3 = 22.0, 3.2, 0.25   # kBq/mL-scale amplitudes
    lam1, lam2, lam3 = 3.1, 0.23, 0.015  # 1/min decay rates

    return scale * (A1*  np.exp(-lam1 * t) +
            A2 *  np.exp(-lam2 * t) +
            A3 *  np.exp(-lam3 * t))



@dataclass
class PBR28Tracer(Tracer):
    scale: float = 1.0
    def aif(self, t_abs: np.ndarray, delta_min: float = 0.0):
        t_rel = np.maximum(t_abs - delta_min, 0.0)
        return pbr28_aif(t_rel, scale = self.scale)

    def true_params(self):
        return dict(
            K1 = 0.18,
            k2 = 0.25,
            k3 = 0.04,
            k4 = 0.01
        )
