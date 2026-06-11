from __future__ import annotations
import numpy as np
from ..base import SeparationAlgorithm, SeparationResult
from ..registry import register
from ..helpers.sequential_nnls import nnls_l2
from ..configs import RecursiveGammaSeparationConfig
@register
class RecursiveUnmix(SeparationAlgorithm):
    '''A simple recursive unmixing algorithm that first fits the first tracer using only
      the early time points, then uses the residual to fit the second tracer, and then iterates
        this process a few times, each time using the residual from the previous step to refine the fits. 
        This is a simple form of recursive separation that can help improve separation when there is significant 
        overlap between the tracers.'''
    name = "recursive_unmix"
    def __init__(self, config:RecursiveGammaSeparationConfig ):
        self.config = config
    def separate(
            self, 
            y: np.ndarray, 
            t_frames: np.ndarray,  
            Phi_1: np.ndarray,
            Phi_2: np.ndarray,
    ) -> SeparationResult: 
        if t_frames is None:
            raise ValueError("RecursiveUnmix requires time frames to determine early time points for initial fit")
        y = np.asarray(y, dtype = float).reshape(-1)
        t_frames = np.asarray(t_frames, dtype = float).reshape(-1)
        Phi_1 = np.asarray(Phi_1, dtype = float)
        Phi_2 = np.asarray(Phi_2, dtype = float)
        
        if y.shape[0] != t_frames.shape[0]: 
            raise ValueError(f"Length of y ({y.shape[0]}) and t_frames ({t_frames.shape[0]}) must match")
        if y.shape[0] != Phi_1.shape[0]: 
            raise ValueError(f"Number of rows in Phi_1 ({Phi_1.shape[0]}) must match length of y ({y.shape[0]})")
        if y.shape[0] != Phi_2.shape[0]:
            raise ValueError(f"Number of rows in Phi_2 ({Phi_2.shape[0]}) must match length of y ({y.shape[0]})")
    
        
        t_cut = float(self.config.t_cut)
        num_iters = self.config.num_iters
        alpha_1 = self.config.alpha_stage_1
        alpha_2 = self.config.alpha_stage_2

        early_mask = t_frames <= t_cut
        if not np.any(early_mask):
            raise ValueError(f"No time points are less than or equal to t_cut ({t_cut}).Please choose a t_cut that includes some early time points.")
        # First fit the first tracer using only the early time points, then use the residual to fit the second tracer,
        # and then iterate this process a few times, each time using the residual from the previous step to refine the fits. 
        # This is a simple form of recursive separation that can help improve separation when there is significant overlap between the tracers.
        print("early_mask any:", np.any(early_mask))
        print("Phi_1 early shape:", Phi_1[early_mask, :].shape)
        print("y early shape:", y[early_mask].shape)

        print("Phi_1 has nan:", np.isnan(Phi_1).any(), "inf:", np.isinf(Phi_1).any())
        print("Phi_2 has nan:", np.isnan(Phi_2).any(), "inf:", np.isinf(Phi_2).any())
        print("y has nan:", np.isnan(y).any(), "inf:", np.isinf(y).any())

        print("Phi_1 early has nan:", np.isnan(Phi_1[early_mask, :]).any(), "inf:", np.isinf(Phi_1[early_mask, :]).any())
        print("y early has nan:", np.isnan(y[early_mask]).any(), "inf:", np.isinf(y[early_mask]).any())

        print("alpha_1:", alpha_1)
        w_1 = nnls_l2(Phi_1[early_mask, :], y[early_mask], alpha = alpha_1)
        est_1 = Phi_1 @ w_1
        resid = np.clip(y - est_1, 0, None)
        w_2 = nnls_l2(Phi_2, resid, alpha = alpha_2)
        est_2 = Phi_2 @ w_2 #tracer 2 estimate
        for i in range(num_iters):
             resid_1 = np.clip(y - est_2, 0, None)
             w_1 = nnls_l2(Phi_1, resid_1, alpha = alpha_1)
             est_3 = Phi_1 @ w_1 #tracer 1 estimate
             resid_2 = np.clip(y - est_3, 0, None)
             w_2 = nnls_l2(Phi_2, resid_2, alpha = alpha_2) 
             est_4 = Phi_2 @ w_2 #tracer 2 estimate
             # Update estimates for next iteration
             est_1 = est_3 #tracer 1 estimate
             est_2 = est_4 #tracer 2 estimate
        return SeparationResult(
            tracer1_curve=est_1,
            tracer2_curve=est_2,
            coef_1=w_1,
            coef_2= w_2,
            metadata={
                "w_1": w_1,
                "w_2": w_2,
                "t_cut": t_cut,
                "early_mask": early_mask,
                "num_iters": num_iters,
                "alpha_1": alpha_1,
                "alpha_2": alpha_2
            },
        )
    
            
