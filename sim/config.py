from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Any
import numpy as np

@dataclass
class TracerConfig:
    fdg_scale: float = 1.0
    pbr_scale: float = 1.0
    fdg_overrides: dict[str, Any] = field(default_factory=dict)
    pbr_overrides: dict[str, Any] = field(default_factory=dict)

from separation.configs import (
    SequentialGammaSeparationConfig,
    RecursiveGammaSeparationConfig,
    JointGammaSeparationConfig,
    TCNLibrarySeparationConfig,
)

from kinetics.configs import (
    GammaBasisConfig,
    InjectionCenteredGammaBasisConfig,
    KineticsInformedLibraryConfig,
)


AlgorithmConfigType = Union[
    SequentialGammaSeparationConfig,
    RecursiveGammaSeparationConfig,
    JointGammaSeparationConfig,
    TCNLibrarySeparationConfig,
]

BasisConfigType = Union[
    GammaBasisConfig,
    InjectionCenteredGammaBasisConfig,
]

@dataclass #check and see if I have an equivalent of this somewhere
class TwoTracerBasisConfig:
    tracer_1_basis: BasisConfigType = field(default_factory=GammaBasisConfig)
    tracer_2_basis: BasisConfigType = field(default_factory=GammaBasisConfig)


@dataclass
class PlotConfig:
    save_nrmse_vs_delay: bool = True
    save_bio_grid: bool = True
    save_measured_grid: bool = True
    save_fdg_aligned_true: bool = True
    save_fdg_aligned_est: bool = True
    save_gamma_plots: bool = True

    tac_grid_ncols: int = 4
    tac_grid_nrows_per_fig: int = 3
    tac_ylim: tuple[float, float] = (0.0, 5.0)
    nrmse_ylim: tuple[float, float] | None = (0.0, 0.6)


@dataclass
class ExperimentConfig:
    experiment_id: str = "fdg_pbr_delay_sweep"
    description: str = "FDG/PBR delay sweep"
    notes: str = ""

    output_root: Path = Path("/Users/maryamhiradfar/Desktop/catana_lab_local/testing_separation_algs_refactored")

    delays_min: np.ndarray = field(default_factory=lambda: np.arange(0, 50, 1))
    frame_edges_min: np.ndarray | None = None
    internal_dt_min: float = 1 / 60.0
    random_seed: int = 42

    # choose one library strategy for the experiment
    tracer: TracerConfig = field(default_factory=TracerConfig)
    gamma_library: TwoTracerBasisConfig | None = None
    kinetics_library: KineticsInformedLibraryConfig | None = None
    # injection_centered_gamma_library: InjectionCenteredGammaBasisConfig | None = None

    algorithm_name: str = "recursive_unmix"
    algorithm_config: AlgorithmConfigType = field(
        default_factory=RecursiveGammaSeparationConfig
    )

    plots: PlotConfig = field(default_factory=PlotConfig)

@dataclass
class ExperimentSummary:
    """
    Summary of a completed experiment run.
    Returned by run_single_experiment.
    """
    experiment_id: str
    run_dir: str
    n_conditions: int

    mean_nrmse_pbr: float
    mean_nrmse_fdg: float

    # optional future fields
    # runtime_sec: float | None = None
    # notes: str | None = None