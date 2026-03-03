from dataclasses import dataclass
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import Any

@dataclass
class GammaLibraryResult: 
    lib_1: np.ndarray
    Gamma_1: np.ndarray
    lib_2: np.ndarray
    Gamma_2: np.ndarray

class GammaLibraryBuilder(ABC):
    name: str
    def __init__(self, **params: Any):
        self.params = params
    @abstractmethod
    def build_library(self, t: np.ndarray, *, context) -> GammaLibraryResult:
        """
        t: time vector (T,)
        kwargs: any additional parameters needed to build the library, e.g. 
            - number of basis functions
            - range of parameters for basis functions
            - scaling factors for each tracer
        """

        raise NotImplementedError
