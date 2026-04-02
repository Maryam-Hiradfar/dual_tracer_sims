# separation/base.py
from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np

@dataclass
class SeparationResult: 
        """ 
        Standardized output across separation methods so downstream
        evaluation and vizualization doesn't care which separation algorithm 
        was used."""
        tracer1_curve: np.ndarray
        tracer2_curve: np.ndarray
        w1: np.ndarray | None = None
        w2: np.ndarray | None = None
        metadata: dict | None = None
class SeparationAlgorithm(ABC): 
    """
    All separation methods must implement this interface
    """
    name: str
    
    @abstractmethod
    def separate(
        self, 
        y: np.ndarray, 
        t_frames: np.ndarray,
        Phi_1: np.ndarray,
        Phi_2: np.ndarray,
    ) -> SeparationResult: 
       raise NotImplementedError
    
        
