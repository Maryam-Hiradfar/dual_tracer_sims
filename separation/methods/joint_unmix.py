from __future__ import annotations
import numpy as np
from typing import Any, Dict, Optional
from .. registry import register
from .. base import SeparationAlgorithm, SeparationResult
from scipy.optimize import nnls
from ..configs import JointGammaSeparationConfig

@register
class JointGammaSeparation(SeparationAlgorithm):
    name  = "joint_unmix_gamma"
    def __init__(self, config:JointGammaSeparationConfig):
        self.config = config
    def separate (
              self, 
              y: np.ndarray, 
              t_frames: np.ndarray,
              Phi_1: np.ndarray, 
              Phi_2: np.ndarray,
 ) -> SeparationResult:
        # Construct the combined library matrix
        Phi = np.concatenate([Phi_1, Phi_2], axis=1)
          
        
        # Solve non-negative least squares to find coefficients for both tracers simultaneously
        w, _ = nnls(Phi, y)
        w_1 = w[:Phi_1.shape[1]]
        w_2 = w[Phi_1.shape[1]:]
        est_1 = Phi_1 @ w_1
        est_2 = Phi_2 @ w_2
        return SeparationResult(
            tracer1_curve=est_1,
            tracer2_curve=est_2,
            coef_1 = w_1,
            coef_2 = w_2,
            metadata={"method": self.name}
         )
