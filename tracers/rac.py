# tracers/rac.py
from dataclasses import dataclass
import numpy as np
from .base import Tracer
from .aif import raclopride_aif, RACLOPRIDE_DEFAULT 

@dataclass
class RacloprideTracer(Tracer):
    scale: float = 1.0
    raclopride_params: dict | None  = None

    def __post_init__(self):
        #Merge user overrides with defaults for params
        base = RACLOPRIDE_DEFAULT.copy()
        if self.raclopride_params is not None: 
            base.update(self.raclopride_params)
        self.raclopride_params = base
        
    def aif(self, t_abs: np.ndarray, delta_min: float = 0.0):
        return raclopride_aif(t_abs)

    def true_params(self):
        return dict(K1=0.09, k2=0.45, k3=0.03, k4=0.04)
