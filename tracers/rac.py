# tracers/rac.py
from dataclasses import dataclass
import numpy as np
from .base import Tracer
from .aif import raclopride_aif

@dataclass
class RacloprideTracer(Tracer):
    def aif(self, t_abs: np.ndarray, delta_min: float = 0.0):
        return raclopride_aif(t_abs)

    def true_params(self):
        return dict(K1=0.09, k2=0.45, k3=0.03, k4=0.04)
