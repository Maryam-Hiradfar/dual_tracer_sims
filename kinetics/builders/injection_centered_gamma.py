#kinetics.injection_centered_gamma.py
from __future__ import annotations

import numpy as np
from ..base import LibraryResult, LibraryBuilder
from ..registry import register
from dataclasses import dataclass
from typing import Any, Dict, Optional
from ..helpers.basis_gamma import build_injection_centered_gamma_library
from ..configs import InjectionCenteredGammaBasisConfig


@register
class InjectionCenteredGammaBuilder(LibraryBuilder): 
    name = "injection_centered_gamma"
    def __init__(self, config:InjectionCenteredGammaBasisConfig):
        self.config = config
    def build_library(self, t:np.ndarray)-> LibraryResult:
        lib, Gamma = build_injection_centered_gamma_library(t, self.config)
        return LibraryResult(lib = lib, Gamma = Gamma)

# @register
# class protocol_informed_gamma(GammaLibraryBuilder):
#     name = "protocol_informed_gamma"
#     def build_library(self, t: np.ndarray, *, context: Optional[Dict[str, Any]] = None) -> GammaLibraryResult:
#         if context is None or not all(k in context for k in ["n_t0_1", "n_tau_1", "n_t0_2", "n_tau_2", "scale_1", "scale_2", "injection_time_1", "injection_time_2", "time_variation_range_1", "time_variation_range_2", "num_variations_1", "num_variations_2"]):
#             raise ValueError("protocol_informed_gamma requires 'n_t0_1', 'n_tau_1', 'n_t0_2', 'n_tau_2', and 'scale_1', 'scale_2', 'injection_time_1', and 'injection_time_2' in context")
#         n_t0_1 = context.get("n_t0_1", 1)
#         n_tau_1 = context.get("n_tau_1", 5)
#         n_t0_2 = context.get("n_t0_2", 1)
#         n_tau_2 = context.get("n_tau_2", 5)
#         scale_1 = context.get("scale_1", 1.0)
#         scale_2 = context.get("scale_2", 1.0)
#         injection_time_1 = context.get("injection_time_1", 0.0)
#         injection_time_2 = context.get("injection_time_2", 0.0)
#         time_variation_range_1 = context.get("time_variation_range_1", 10.0)
#         time_variation_range_2 = context.get("time_variation_range_2",10.0)     
#         num_variations_1 = context.get("num_variations_1", 5)
#         num_variations_2 = context.get("num_variations_2", 5)
#         # For protocol-informed gamma, we can set the t0 values to be around the injection times for each tracer, and set the tau values to cover a range of plausible kinetics. 
#         #Additionally, we vary the t0 values around the injection time to account for uncertainty in the exact timing of the bolus arrival. 
#         # The scale can also be set based on calculated peaks for each tracer, or based on expected relative uptake.

#         lib_1, Gamma_1, lib_2, Gamma_2 = build_gamma_library_for_two_tracers(
#             t, n_t0_1, n_tau_1, n_t0_2, n_tau_2, scale_1, scale_2
#         )
#         for i in range(num_variations_1):
#             for j in range(num_variations_2):
#                 lib_1_temp, Gamma_1_temp, lib_2_temp, Gamma_2_temp = build_gamma_library_for_two_specific_injection_time_points(
#                     t, [injection_time_1 + i - time_variation_range_1//2], n_tau_1, [injection_time_2 + j - time_variation_range_2//2], n_tau_2, scale_1, scale_2
#                 )
#                 lib_1 = np.append(lib_1_temp)
#                 lib_2 = np.append(lib_2_temp)
        
#         Gamma_1 = lib_1.T
#         Gamma_2 = lib_2.T

#         return GammaLibraryResult(lib_1=lib_1, Gamma_1=Gamma_1, lib_2=lib_2, Gamma_2=Gamma_2)