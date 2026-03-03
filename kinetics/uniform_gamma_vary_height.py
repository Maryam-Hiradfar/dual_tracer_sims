#uniform_gamma_vary_height.py
import numpy as np
from .base import GammaLibraryBuilder, GammaLibraryResult
from .registry import register
from dataclasses import dataclass
from __future__ import annotations
from typing import Any, Dict, Optional
from .basis_gamma import build_gamma_library_for_two_tracers


@register
class uniform_gamma_vary_height(GammaLibraryBuilder):
    name = "uniform_gamma_vary_height"
    def build_library(self, t: np.ndarray, *, context: Optional[Dict[str, Any]] = None) -> GammaLibraryResult:
        if context is None or not all(k in context for k in ["n_t0_1", "n_tau_1", "n_t0_2", "n_tau_2", "scale_1", "scale_2", "variation_factor_upper", "variation_factor_lower", "num_variations"]):
            raise ValueError("uniform_gamma_fixed_height requires 'n_t0_1', 'n_tau_1', 'n_t0_2', 'n_tau_2', and 'scale_1', and 'scale_2' in context")
        n_t0_1 = context.get("n_t0_1", 5)
        n_tau_1 = context.get("n_tau_1", 5)
        n_t0_2 = context.get("n_t0_2", 5)
        n_tau_2 = context.get("n_tau_2", 5)
        scale_1 = context.get("scale_1", 1.0)
        scale_2 = context.get("scale_2", 1.0)
        variation_factor_upper = context.get("variation_factor_upper", 2.0)
        variation_factor_lower = context.get("variation_factor_lower", 0.5)
        num_variations = context.get("num_variations", 5)
        for i in range(num_variations):
            scale_1 *= np.random.uniform(variation_factor_lower, variation_factor_upper)
            scale_2 *= np.random.uniform(variation_factor_lower, variation_factor_upper)
        
        lib_1 = []
        Gamma_1 = []
        lib_2 = []
        Gamma_2 = []
        for i in range(num_variations):
            lib_1_temp, Gamma_1_temp, lib_2_temp, Gamma_2_temp = build_gamma_library_for_two_tracers(
                t, n_t0_1, n_tau_1, n_t0_2, n_tau_2, scale_1, scale_2
            )
            lib_1.append(lib_1_temp)
            lib_2.append(lib_2_temp)
        Gamma_1 = lib_1.T
        Gamma_2 = lib_2.T

        return GammaLibraryResult(lib_1=lib_1, Gamma_1=Gamma_1, lib_2=lib_2, Gamma_2=Gamma_2)
        
