from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from typing import Any

@dataclass
class LibraryResult:
    lib: np.ndarray #shape: (n_basis, n_time_points)
    Gamma: np.ndarray #shape: (n_time_points, n_basis)

class LibraryBuilder(ABC): 
    name: str
    @abstractmethod
    def build_library(self, t:np.ndarray) -> LibraryResult:
        raise NotImplementedError
    
