from __future__ import annotations
import numpy as np
from typing import Any, Dict, Optional
from .base import SeparationAlgorithm, SeparationResult
from .registry import register
from scipy.optimize import nnls
from .sequential_nnls import nnls_l2
@register
class RecursiveUnmix(SeparationAlgorithm):
    '''regularization happens only for the first estimates before recursive refinement,
    and then the subsequent refinements are just non-negative least squares without regularization,
    which can help improve separation when there is significant overlap between the tracers.'''
    name = "recursive_unmix"
    def separate(
            self, 
            y: np.ndarray, 
            x: np.ndarray, 
            t: np.ndarray, 
            *, 
            context: Optional[Dict[str, Any]] = None, 
    ) -> SeparationResult: 
        if context is None or "Phi_1" not in context or "Phi_2" not in context: 
                raise ValueError("RecursiveUnmix requires 'Phi_1' and 'Phi_2' in context")
        Phi_1 = context.get("Phi_1")
        Phi_2 = context.get("Phi_2")
        num_iters = context.get("num_iters", 5)
        alpha_val = context.get("alpha", 0.1)
        if Phi_1 is None or Phi_2 is None:
            raise ValueError("RecursiveUnmix requires 'Phi_1' and 'Phi_2' in context")
        t_frames = np.asarray(context.get("t_frames", t), dtype=float).reshape(-1)
        t_cut_raw = context.get("t_cut", float(np.median(t_frames)))
        t_cut = float(np.asarray(t_cut_raw).reshape(-1)[0]) #ensure t_cut is a scalar
        early_mask = t_frames <= t_cut
        # First fit the first tracer using only the early time points, then use the residual to fit the second tracer,
        # and then iterate this process a few times, each time using the residual from the previous step to refine the fits. 
        # This is a simple form of recursive separation that can help improve separation when there is significant overlap between the tracers.
        w_1 = nnls_l2(Phi_1[early_mask, :], y[early_mask], alpha = alpha_val)
        est_1 = Phi_1 @ w_1
        resid = np.clip(y - est_1, 0, None)
        w_2 = nnls_l2(Phi_2, resid, alpha = alpha_val)
        est_2 = Phi_2 @ w_2 #tracer 2 estimate
        for i in range(num_iters):
             resid_1 = np.clip(y - est_2, 0, None)
             w_3, _ = nnls(Phi_1, resid_1)
             est_3 = Phi_1 @ w_3 #tracer 1 estimate
             resid_2 = np.clip(y - est_3, 0, None)
             w_4, _ = nnls(Phi_2, resid_2) 
             est_4 = Phi_2 @ w_4 #tracer 2 estimate
             # Update estimates for next iteration
             est_1 = est_3 #tracer 1 estimate
             est_2 = est_4 #tracer 2 estimate
        return SeparationResult(
            tracer1_curve=est_1,
            tracer2_curve=est_2,
            extras={
                "w_1": w_1,
                "w_2": w_2,
                "t_cut": t_cut,
                "early_mask": early_mask,
                "num_iters": num_iters,
                "alpha": alpha_val
            }
        )
    
            
