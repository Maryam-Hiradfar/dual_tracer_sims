from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict
from .tac_simulator import simulate_clean_dual_tac
from core.protocol import Protocol
from tracers import tracer_factory
import numpy as np
@dataclass
class HybridSample: 
    y_meas: np.ndarray 
    y_tracer1_true: np.ndarray #non-decayed
    y_tracer2_true: np.ndarray #non-decayed
    y_tracer1_phys: np.ndarray #decayed
    y_tracer2_phys: np.ndarray #decayed
    metadata: Dict[str, Any] = field(default_factory = dict)
@dataclass 
class SampleSpec: 
    tracer1_name: str
    tracer2_name: str
    tracer1_scale: float
    tracer2_scale: float
    delta_min: float
    tracer1_first: bool
    seed:int | None = None
    # tracer1_overrides: dict | None = None
    # tracer2_overrides: dict | None = None
@dataclass
class DataSetConfig: 
    n_sample: int
    delta_choices_min: list[float]
    scale_range: tuple[float, float]
    randomize_order: bool = True

class SampleGenerator: 
    def __init__(self, timegrid, tracer1, tracer2, protocol, rng = None):
        self.timegrid = timegrid
        self.tracer1 = tracer1
        self.protocol = protocol
        self.tracer2 = tracer2
        self.rng = rng
    def generate_sample(self) -> HybridSample: 
        sim = simulate_clean_dual_tac(
            timegrid=self.timegrid,
            tracer1 = self.tracer1, 
            tracer2 = self.tracer2, 
            protocol = self.protocol,
            rng = self.rng,
        )

        return HybridSample(
            y_meas = sim.y_meas,
            y_tracer1_true = sim.Ct1_bio,
            y_tracer2_true = sim.Ct2_bio,
            y_tracer1_phys = sim.Ct1_phys,
            y_tracer2_phys = sim.Ct2_phys,
            metadata = {
                "delta_min": sim.delta_min,
                # "tracer1_name": sim.tracer1.name,
                # "tracer2_name": sim.tracer2_name,
                "lam1": sim.lam1,
                "lam2": sim.lam2,
            }
        )
        
class DatasetGenerator: 
    def __init__(self, timegrid, tracer_factory, config: DataSetConfig, seed = None):
        self.timegrid = timegrid
        self.tracer_factory = tracer_factory
        self.config = config
        self.rng = np.random.default_rng(seed)

    def sample_spec(self)-> SampleSpec:
        if self.config.randomize_order: 
             tracer1_first = self.rng.choice([True, False])
        else: 
            tracer1_first = True
        scale_1 = self.rng.uniform(*self.config.scale_range)
        scale_2 = self.rng.uniform(*self.config.scale_range)
        delta_min = self.rng.choice(self.config.delta_choices_min)

        if tracer1_first: # in the future we will make tracer1 and tracer2 to be configurable and not limited to PBR28 and FDG
            tracer1, tracer2 = "PBR28", "FDG"
        else: 
            tracer1, tracer2 = "FDG", "PBR28"

        # overrides_1 = self.sample_overrides(tracer1)
        # overrides_2 = self.sample_overrides(tracer2)
        return SampleSpec(
            tracer1_name= tracer1, 
            tracer2_name= tracer2,
            tracer1_scale = scale_1,
            tracer2_scale = scale_2,
            delta_min  = delta_min,
            tracer1_first = tracer1_first,
            # tracer1_overrides= overrides_1,
            # tracer2_overrides=overrides_2,
        )

    
    def make_sample_generator(self, spec:SampleSpec) -> SampleGenerator:
        tracer1 = self.tracer_factory(spec.tracer1_name, scale = spec.tracer1_scale)
        tracer2 = self.tracer_factory(spec.tracer2_name, scale = spec.tracer2_scale)
        protocol = Protocol(delta_min = spec.delta_min)
        return SampleGenerator(
            timegrid = self.timegrid,
            tracer1= tracer1, 
            tracer2 = tracer2, 
            protocol = protocol, 
            rng = self.rng,
        )
    
    def generate_dataset(self):
        samples = []
        specs = []
        n = self.config.n_sample
        for _ in range(n): 
            spec = self.sample_spec()
            gen = self.make_sample_generator(spec)
            sample = gen.generate_sample()
            samples.append(sample)
            specs.append(spec)
        return samples, specs


    
        
