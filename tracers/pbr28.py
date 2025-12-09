# tracers/pbr28.py
from dataclasses import dataclass
import numpy as np
from .base import Tracer
from .aif import pbr28_aif



@dataclass
class PBR28Tracer(Tracer):
    scale: float = 1.0
    pbr28_params: dict | None  = None

    def __post_init__(self):
        #Merge user overrides with defaults for params
        base = PBR28_DEFAULT.copy()
        if self.pbr28_params is not None: 
            base.update(self.pbr28_params)
        self.pbr28_params = base
        
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
