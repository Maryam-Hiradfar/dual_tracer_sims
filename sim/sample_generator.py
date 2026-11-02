from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict
from .tac_simulator import simulate_dual_tracer_measurement
from core.protocol import Protocol
from tracers import tracer_factory
import numpy as np
@dataclass
class HybridSample:
    y_meas_tavg_n1_d1: np.ndarray
    y_clean_tavg_n0_d1: np.ndarray
    y_tracer1_true_tavg_n0_d0: np.ndarray #non-decayed
    y_tracer2_true_tavg_n0_d0: np.ndarray #non-decayed
    y_tracer1_phys_tavg_n0_d1: np.ndarray #decayed
    y_tracer2_phys_tavg_n0_d1: np.ndarray #decayed
    expected_counts: np.ndarray
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
    first_tracer_name: str = "PBR28"
    second_tracer_name: str = "FDG"
    randomize_order: bool = True

class SampleGenerator:
    def __init__(self, timegrid, tracer1, tracer2, protocol, rng = None):
        self.timegrid = timegrid
        self.tracer1 = tracer1
        self.protocol = protocol
        self.tracer2 = tracer2
        self.rng = rng
    def generate_sample(self) -> HybridSample:
        sim = simulate_dual_tracer_measurement(
            timegrid=self.timegrid,
            tracer1 = self.tracer1,
            tracer2 = self.tracer2,
            protocol = self.protocol,
            rng = self.rng,
        )

        return HybridSample(
            y_meas_tavg_n1_d1 = sim.y_meas_tavg_n1_d1,
            y_clean_tavg_n0_d1 = sim.y_clean_tavg_n0_d1,
            y_tracer1_true_tavg_n0_d0 = sim.Ct1_bio_tavg_n0_d0,
            y_tracer2_true_tavg_n0_d0 = sim.Ct2_bio_tavg_n0_d0,
            y_tracer1_phys_tavg_n0_d1 = sim.Ct1_phys_tavg_n0_d1,
            y_tracer2_phys_tavg_n0_d1 = sim.Ct2_phys_tavg_n0_d1,
            expected_counts = sim.expected_counts,
            metadata = {
                "delta_min": sim.delta_min,
                "tracer1_name": self.tracer1.name,
                "tracer2_name": self.tracer2.name,
                "tracer1_scale": self.tracer1.scale,
                "tracer2_scale": self.tracer2.scale,
                "lam1": sim.lam1,
                "lam2": sim.lam2,
            }
        )

class DatasetGenerator:
    def __init__(self, timegrid, tracer_factory, config: DataSetConfig, seed = None):
        if config.first_tracer_name == config.second_tracer_name:
            raise ValueError("first and second tracer names must be different")
        self.timegrid = timegrid
        self.tracer_factory = tracer_factory
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def sample_spec(self)-> SampleSpec:
        if self.config.randomize_order:
             tracer1_first = bool(self.rng.choice([True, False]))
        else:
            tracer1_first = True
        scale_1 = float(self.rng.uniform(*self.config.scale_range))
        scale_2 = float(self.rng.uniform(*self.config.scale_range))
        delta_min = float(self.rng.choice(self.config.delta_choices_min))
        sample_seed = int(
            self.rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)
        )

        if tracer1_first:
            tracer1 = self.config.first_tracer_name
            tracer2 = self.config.second_tracer_name
        else:
            tracer1 = self.config.second_tracer_name
            tracer2 = self.config.first_tracer_name

        # overrides_1 = self.sample_overrides(tracer1)
        # overrides_2 = self.sample_overrides(tracer2)
        return SampleSpec(
            tracer1_name= tracer1,
            tracer2_name= tracer2,
            tracer1_scale = scale_1,
            tracer2_scale = scale_2,
            delta_min  = delta_min,
            tracer1_first = tracer1_first,
            seed=sample_seed,
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
            rng=np.random.default_rng(spec.seed),
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
