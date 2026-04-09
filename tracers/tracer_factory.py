import numpy as np
from tracers.fdg import FDGTracer
from tracers.pbr28 import PBR28Tracer
def tracer_factory(tracer_name:str, scale: float = 1.0, overrides = None): 
    if tracer_name == "FDG": 
        return FDGTracer(
            name = "FDG",
            half_life_min= 109.8, 
            scale = scale, 
            feng_params = overrides if overrides else None,
            )
    elif tracer_name == "PBR28":
        return PBR28Tracer(
            name = "PBR28", 
            half_life_min= 20.4,
            scale = scale, 
            pbr28_params= overrides if overrides else None,
        )
    else:
        raise ValueError(f"unknown tracer: {tracer_name}")