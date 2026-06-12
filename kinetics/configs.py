from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

#----------------------------------------
# Small Reusable Config Pieces
#----------------------------------------

@dataclass
class TwoTCMParams: 
    K1: float
    k2: float
    k3: float
    k4: float
@dataclass 
class ParamSweep1D: 
    low: float
    high: float
    num : int  = 1
@dataclass 
class TwoTCMSweepConfig: 
    K1: ParamSweep1D
    k2: ParamSweep1D
    k3: ParamSweep1D
    k4: ParamSweep1D


@dataclass   
class InjectionTimeConfig: 
    """
    Configuration for varying injection/start time around a nominal injection time 
    """
    injection_time: float = 0.0
    variation_range: float = 10.0
    num_variations: int = 5

    def validate(self) -> None: 
        if self.num_variations < 1:
            raise ValueError("num_variations must be at least 1")
        if self.variation_range < 0:
            raise ValueError("variation_range must be non-negative")
    
@dataclass
class KineticsInformedTracerConfig: 
    tracer_name : str
    nominal_params: TwoTCMParams
    injection: InjectionTimeConfig
    sweep: TwoTCMSweepConfig
    scale: float = 1.0

#----------------------------------------
#  basis library configs
#----------------------------------------

@dataclass 
class KineticsInformedLibraryConfig: 
    tracer_1: KineticsInformedTracerConfig
    tracer_2: KineticsInformedTracerConfig

    
@dataclass
class GammaBasisConfig: 
    """
    Configuration for gamma basis library for one tracer
    """
    n_t0: int = 5
    n_tau: int = 5
    scale: float = 1.0

    t0_min : float = 0.0
    t0_max_offset: float = 10.0 #max t0 = t[-1]  - t0_max_offset
    tau_min: float = 0.3
    tau_max: float = 40.0

    def validate(self) -> None: 
        if self.n_t0 <1: 
            raise ValueError("n_t0 must be at least 1")
        if self.n_tau <1:
            raise ValueError("n_tau must be at least 1")
        if self.tau_min <= 0 or self.tau_max <= 0:
            raise ValueError("tau_min and tau_max must be positive")
        if self.tau_min >= self.tau_max:
            raise ValueError("tau_min must be less than tau_max")
        if self.t0_max_offset < 0:
            raise ValueError("t0_max_offset must be positive")
        
@dataclass 
class InjectionCenteredGammaBasisConfig:
    """
    Gamma basis config for one tracer where t0 values are centered
    around a nominal injection time rather than spread uniformly
    over the whole scan.
    """
    injection_time: float = 0.0
    variation_range_after: float = 5.0
    variation_range_before: float = 5.0
    num_t0: int = 5

    n_tau: int = 5
    scale: float = 1.0

    tau_min: float = 0.3
    tau_max: float = 40.0

    def validate(self) -> None:
        if self.num_t0 < 1:
            raise ValueError("num_t0 must be at least 1")
        if self.n_tau < 1:
            raise ValueError("n_tau must be at least 1")
        if self.variation_range_before < 0:
            raise ValueError("variation_range must be non-negative")
        if self.variation_range_after < 0:
            raise ValueError("variation_range must be non-negative")
        if self.tau_min <= 0 or self.tau_max <= 0:
            raise ValueError("tau_min and tau_max must be positive")
        if self.tau_min >= self.tau_max:
            raise ValueError("tau_min must be less than tau_max")

#----------------------------------------
# factory functions
#----------------------------------------

def make_fdg_tracer_config() -> KineticsInformedTracerConfig: 
    return KineticsInformedTracerConfig(
        tracer_name="FDG",
        nominal_params=TwoTCMParams(K1=0.12, k2=0.25, k3=0.06, k4=0.00),
        injection=InjectionTimeConfig(injection_time=0.0, variation_range=10.0, num_variations=5),
        sweep=TwoTCMSweepConfig(
            K1=ParamSweep1D(low=0.08, high=0.16, num=5),
            k2=ParamSweep1D(low=0.20, high=0.30, num=5),
            k3=ParamSweep1D(low=0.05, high=0.08, num=5),
            k4=ParamSweep1D(low=0.00, high=0.005, num=5)
        ),
        scale = 1.0
    )
def make_pbr_tracer_config() -> KineticsInformedTracerConfig: 
    return KineticsInformedTracerConfig(
        tracer_name="PBR",
        nominal_params=TwoTCMParams(K1 = 0.18,
            k2 = 0.25,
            k3 = 0.04,
            k4 = 0.01),
        injection=InjectionTimeConfig(injection_time=0.0, variation_range=10.0, num_variations=5),
        sweep=TwoTCMSweepConfig(
            K1=ParamSweep1D(low=0.16, high=0.20, num=5),
            k2=ParamSweep1D(low=0.20, high=0.30, num=5),
            k3=ParamSweep1D(low=0.02, high=0.06, num=5),
            k4=ParamSweep1D(low=0.00, high=0.01, num=5)
        ),
        scale = 1.0
    )
