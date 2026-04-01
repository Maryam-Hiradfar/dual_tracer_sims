from dataclasses import dataclass
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import Any

@dataclass
class GammaLibraryResult:
    lib: np.ndarray #shape: (n_basis, n_time_points)
    Gamma: np.ndarray #shape: (n_time_points, n_basis)

class GammaLibraryBuilder(ABC): 
    name: str
    @abstractmethod
    def build_library(self, t:np.ndarray) -> GammaLibraryResult:
        raise NotImplementedError
    
