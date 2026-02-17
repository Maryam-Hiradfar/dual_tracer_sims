from __future__ import annotations
import numpy as np
from typing import Any, Dict, Optional
from .base import SeparationAlgorithm, SeparationResult
from .registry import register
from scipy.optimize import nnls

@register
class joint_unmix(SeparationAlgorithm):
    name  = "joint_unmix"
    def separate (
            self, 
            y:np.ndarray, 
            t: np.ndarray, 
            *, 
            context: Optional[Dict[str, Any]] = None, 
    ) -> SeparationResult:
        if context is None or "Phi_1" not in context or "Phi_2" not in context:
                raise ValueError("JointUnmix requires 'phi_1' and 'phi_2' in context")
        Phi_1 = context.get("Phi_1")
        Phi_2 = context.get("Phi_2")
        if Phi_1 is None or Phi_2 is None:
            raise ValueError("JointUnmix requires 'Phi_1' and 'Phi_2' in context")
        # Stack the basis functions for both tracers together and solve for all coefficients at once
        Phi = np.hstack([Phi_1, Phi_2])  
        
        # Solve non-negative least squares to find coefficients for both tracers simultaneously
        w, _ = nnls(Phi, y)
        w_1 = w[:Phi_1.shape[1]]
        w_2 = w[Phi_1.shape[1]:]
        est_1 = Phi_1 @ w_1
        est_2 = Phi_2 @ w_2
        return SeparationResult(
            tracer1_curve=est_1,
            tracer2_curve=est_2,
            extras={
                "w_1": w_1,
                "w_2": w_2
            }
          
    )
