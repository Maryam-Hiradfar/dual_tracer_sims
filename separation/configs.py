# separation/configs.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SequentialGammaSeparationConfig:
    alpha_stage_1: float = 1e-3
    alpha_stage_2: float = 1e-3
    t_cut : float = 20.0
    positive_only: bool = True

@dataclass
class RecursiveGammaSeparationConfig: 
    alpha_stage_1: float = 1e-3
    alpha_stage_2: float = 1e-3
    t_cut : float = 20.0
    num_iters: int = 5
    positive_only: bool = True
    
@dataclass
class JointGammaSeparationConfig: 
    alpha: float = 1e-3
    positive_only: bool = True

@dataclass
class TCNLibrarySeparationConfig: 
    checkpoint_path: str
    normalize_input: bool = True
    device: str = "cpu"  # or "cuda" if GPU is available
