from __future__ import annotations
import numpy as np
from typing import Any, Dict, Optional
from .base import SeparationAlgorithm, SeparationResult
from .registry import register
from scipy.optimize import nnls
from sequential_unmix import nnls_l2

@register
class sequential_unmix_l2(SeparationAlgorithm):
    name = "sequential_unmix_l2"
    def separate(
            self, 
            y = np.ndarray, 
            t = np.ndarray, 
            *, 
            context: Optional[Dict[str, Any]] = None, 
    ) -> SeparationResult:
        if context is None or "Phi_1" not in context or "Phi_2" not in context:
                raise ValueError("SequentialUnmixL2 requires 'Phi_1' and 'Phi_2' in context")
        Phi_1 = context.get("Phi_1")
        Phi_2 = context.get("Phi_2")
        if Phi_1 is None or Phi_2 is None:
            raise ValueError("SequentialUnmixL2 requires 'Phi_1' and 'Phi_2' in context")
        t_frames = context.get("t_frames", t)
        t_cut = context.get("t_cut", np.median(t_frames))
        alpha = context.get("alpha", 1.0)
        early_mask = t_frames <= t_cut
        #sequentially separate the signals, with L2 regularization on each step separately
        w_1 = nnls_l2(Phi_1[early_mask, :], y[early_mask], alpha)
        w_2 = nnls_l2(Phi_2, np.clip(y - Phi_1 @ w_1, 0, None), alpha)
        est_1 = Phi_1 @ w_1
        est_2 = Phi_2 @ w_2
        return SeparationResult(
            tracer1_curve=est_1,
            tracer2_curve=est_2,
            extras={
                "w_1": w_1,
                "w_2": w_2,
                "t_cut": t_cut,
                "alpha": alpha,
                "early_mask": early_mask
            }
            


    )