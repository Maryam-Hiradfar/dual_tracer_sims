#kinetics.kinetic_informed_library.py
import numpy as np
from .base import GammaLibraryBuilder, GammaLibraryResult
from .registry import register
from dataclasses import dataclass
from __future__ import annotations
from typing import Any, Dict, Optional
from .basis_gamma import build_gamma_library_for_two_tracers

@register
class kinetic_informed_library(GammaLibraryBuilder):
    name = "kinetic_informed_library"
    def build_library(self, t: np.ndarray, *, context: Optional[Dict[str, Any]] = None) -> GammaLibraryResult:
        if context is None or not all(k in context for k in ["n_t0_1", "n_tau_1", "n_t0_2", "n_tau_2", "scale_1", "scale_2"]):
            raise ValueError("kinetic_informed_library requires 'n_t0_1', 'n_tau_1', 'n_t0_2', 'n_tau_2', and 'scale_1', and 'scale_2' in context")
        n_t0_1 = context.get("n_t0_1", 5)
        n_tau_1 = context.get("n_tau_1", 5)
        n_t0_2 = context.get("n_t0_2", 5)
        n_tau_2 = context.get("n_tau_2", 5)
        scale_1 = context.get("scale_1", 1.0)
        scale_2 = context.get("scale_2", 1.0)

        lib_1, Gamma_1, lib_2, Gamma_2 = build_gamma_library_for_two_tracers(
            t, n_t0_1, n_tau_1, n_t0_2, n_tau_2, scale_1, scale_2
        )
        return GammaLibraryResult(lib_1=lib_1, Gamma_1=Gamma_1, lib_2=lib_2, Gamma_2=Gamma_2)