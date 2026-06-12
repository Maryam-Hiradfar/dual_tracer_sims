# kinetics/helpers/twotcm.py
#Incomplete and in development (06/12/2026) - Maryam Hiradfar
from __future__ import annotations

import numpy as np

from ..configs import KineticsInformedTracerConfig
from kinetics.models.twotcm import simulate_2tcm
from tracers.aif import feng_aif, pbr28_aif

def get_aif_func(tracer_name: str):
    tracer_name = tracer_name.upper()
    if tracer_name == "FDG":
        return feng_aif
    if tracer_name in {"PBR", "PBR28"}:
        return pbr28_aif
    raise ValueError(f"Unsupported tracer_name: {tracer_name}")


def build_two_tcm_library(t: np.ndarray, config: KineticsInformedTracerConfig):
    config.injection.validate()

    aif_func = get_aif_func(config.tracer_name)

    inj = config.injection
    t0_low = max(0.0, inj.injection_time - inj.variation_range)
    t0_high = inj.injection_time + inj.variation_range
    t0_values = np.linspace(t0_low, t0_high, inj.num_variations)

    sweep = config.sweep
    K1_values = np.linspace(sweep.K1.low, sweep.K1.high, sweep.K1.num)
    k2_values = np.linspace(sweep.k2.low, sweep.k2.high, sweep.k2.num)
    k3_values = np.linspace(sweep.k3.low, sweep.k3.high, sweep.k3.num)
    k4_values = np.linspace(sweep.k4.low, sweep.k4.high, sweep.k4.num)

    lib = []
    for t0 in t0_values:
        for K1 in K1_values:
            for k2 in k2_values:
                for k3 in k3_values:
                    for k4 in k4_values:
                        params = dict(K1=K1, k2=k2, k3=k3, k4=k4)
                        tac = simulate_2tcm(t, aif_func, t0, params)
                        lib.append(config.scale * tac)

    lib = np.asarray(lib)
    return lib, lib.T