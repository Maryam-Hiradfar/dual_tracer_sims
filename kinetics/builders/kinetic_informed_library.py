#kinetics.kinetic_informed_library.py
#Incomplete and in development (06/12/2026) - Maryam Hiradfar
import numpy as np
from ..base import LibraryBuilder, LibraryResult
from ..registry import register
from dataclasses import dataclass
from __future__ import annotations
from typing import Any, Dict, Optional
from ..helpers.basis_gamma import build_gamma_library_for_two_tracers
from kinetics.helpers.twotcm import simulate_2tcm
from ..configs import KineticsInformedLibraryConfig
from ..helpers.twotcm import build_two_tcm_library

@register 
class KineticInformedLibraryBuilder(LibraryBuilder):
    name = "kinetic_informed_library"
    def __init__ (self, config:KineticsInformedLibraryConfig):
        self.config = config
    def build_library(self, t:np.ndarray) -> LibraryResult:
        lib, Gamma = build_two_tcm_library(t, self.config)
        return LibraryResult(lib = lib, Gamma = Gamma)
# @register
# class kinetic_informed_library(GammaLibraryBuilder):
#     name = "kinetic_informed_library"
#     def build_library(self, t: np.ndarray, *, context: Optional[Dict[str, Any]] = None) -> GammaLibraryResult:
#         if context is None or not all(k in context for k in ["tracer_1", "tracer_2", "n_t0_1", "n_tau_1", "n_t0_2", "n_tau_2", "dose_1", "dose_2",  
#                                                              "injection_time_1", "injection_time_2", "time_variation_range_1", 
#                                                              "time_variation_range_2", "num_variations_1", "num_variations_2"]):
#             raise ValueError("kinetic_informed_library requires 'n_t0_1', 'n_tau_1', 'n_t0_2', 'n_tau_2', and 'dose_1', and 'dose_2' in context")
#         n_t0_1 = context.get("n_t0_1", 5)
#         n_tau_1 = context.get("n_tau_1", 5)
#         n_t0_2 = context.get("n_t0_2", 5)
#         n_tau_2 = context.get("n_tau_2", 5)
#         injection_time_1 = context.get("injection_time_1", 0.0)
#         injection_time_2 = context.get("injection_time_2", 0.0)
#         time_variation_range_1 = context.get("time_variation_range_1", 10.0)
#         time_variation_range_2 = context.get("time_variation_range_2",10.0)     
#         num_variations_1 = context.get("num_variations_1", 5)
#         num_variations_2 = context.get("num_variations_2", 5)
#         dose_1 = context.get("dose_1", 1.0)
#         dose_2 = context.get("dose_2", 1.0)

#         #I don't think the scales should matter for the gamma libraries...
#         lib_1, Gamma_1, lib_2, Gamma_2 = build_gamma_library_for_two_tracers(
#             t, n_t0_1, n_tau_1, n_t0_2, n_tau_2, scale_1=1.0, scale_2=1.0
#         )
#         return GammaLibraryResult(lib_1=lib_1, Gamma_1=Gamma_1, lib_2=lib_2, Gamma_2=Gamma_2)