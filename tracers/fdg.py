# tracers/fdg.py
from dataclasses import dataclass
import numpy as np
from .base import Tracer
from .aif import feng_aif, FENG_FDG_DEFAULT

@dataclass
class FDGTracer(Tracer):
    scale: float = 1.0
    feng_params: dict | None  = None
    def __post_init__(self):
        #Merge user overrides with defaults for params
        base = FENG_FDG_DEFAULT.copy()
        if self.feng_params is not None: 
            base.update(self.feng_params)
        self.feng_params = base


    def aif(self, t_abs: np.ndarray, delta_min: float = 0.0):
        t_rel = np.maximum(t_abs - delta_min, 0.0)
        return feng_aif(t_rel, **self.feng_params, scale = self.scale)

    def true_params(self):
        return dict(K1=0.12, k2=0.25, k3=0.06, k4=0.00)
