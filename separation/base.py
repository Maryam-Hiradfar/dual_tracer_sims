# separation/base.py
from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod, field
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
        extras: Dict[str, Any] = field(default_factory=dict)
class SeparationAlgorithm(ABC): 
    """
    All separation methods must implement this interface
    """
    name: str
    def __init__(self, **params: Any):
        self.params = params
    @abstractmethod
    def separate(
        self, 
        y: np.ndarray,
        t: np.ndarray,
        *,
        context: Optional[Dict[str, Any]] = None, 
    ) -> SeparationResult: 
        """
        y: measured mixture curve (T, )
        t: time vector (T,)
        context: optional dict that can contain any additional information needed for separation, e.g. 
            - tracer AIFs
            - basis functions
            - protocol info (e.g. early/late cut times)
         """
        raise NotImplementedError
    
        
