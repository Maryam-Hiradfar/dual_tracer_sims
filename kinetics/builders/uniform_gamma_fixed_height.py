#kinetics.uniform_gamma_fixed_height.py
import numpy as np
from ..base import GammaLibraryBuilder, LibraryResult
from ..registry import register
from dataclasses import dataclass
from __future__ import annotations
from typing import Any, Dict, Optional
from ..helpers.basis_gamma import build_gamma_library_for_two_tracers
from ..configs import GammaBasisConfig
from ..helpers.basis_gamma import build_uniform_gamma_library



@register
class UniformGammaBuilder(GammaLibraryBuilder):
    name = "uniform_gamma"

    def __init__(self, config:GammaBasisConfig):
        self.config = config
    def build_library(self, t:np.ndarray)-> LibraryResult:
        lib, Gamma = build_uniform_gamma_library(t, self.config)
        return LibraryResult(lib = lib, Gamma = Gamma)


        
