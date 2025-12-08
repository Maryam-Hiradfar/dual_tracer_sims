# tracers/fdg.py
from dataclasses import dataclass
import numpy as np
from .base import Tracer
from .aif import feng_aif

@dataclass
class FDGTracer(Tracer):
    scale: 1.0
    feng_params: dict = None

    def aif(self, t_abs: np.ndarray, delta_min: float = 0.0):
        t_rel = np.maximum(t_abs - delta_min, 0.0)
        return feng_aif(t_rel, **self.feng_params, scale = self.scale)

    def true_params(self):
        return dict(K1=0.12, k2=0.25, k3=0.06, k4=0.00)
