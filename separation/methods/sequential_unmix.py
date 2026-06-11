from __future__ import annotations
import numpy as np
from typing import Any, Dict, Optional
from ..base import SeparationAlgorithm, SeparationResult
from ..registry import register
from scipy.optimize import nnls
from ..helpers.sequential_nnls import nnls_l2

@register
class SequentialUnmix(SeparationAlgorithm):
    name = "sequential_unmix"
    def __init__(self, config):
        self.config = config
    def separate(
            self, 
            y : np.ndarray, 
            t_frames : np.ndarray, 
            Phi_1 : np.ndarray,
            Phi_2 : np.ndarray,
            ) -> SeparationResult:
        
        t_cut = self.config.t_cut
        alpha_1 = self.config.alpha_stage_1
        alpha_2 = self.config.alpha_stage_2
        y = np.asarray(y).reshape(-1) #ensure y is a 1d array
        t_frames = np.asarray(t_frames).reshape(-1) #ensure t_frames is a 1d array
        Phi_1 = np.asarray(Phi_1, dtype = float)
        Phi_2 = np.asarray(Phi_2, dtype = float)
        if y.shape[0] != t_frames.shape[0]: 
            raise ValueError(f"Length of y ({y.shape[0]}) and t_frames ({t_frames.shape[0]}) must match")
        if y.shape[0] != Phi_1.shape[0]:
            raise ValueError(f"Number of rows in Phi_1 ({Phi_1.shape[0]}) must match length of y ({y.shape[0]})")
        if y.shape[0] != Phi_2.shape[0]:
            raise ValueError(f"Number of rows in Phi_2 ({Phi_2.shape[0]}) must match length of y ({y.shape[0]})")
        
    
        early_mask = t_frames <= t_cut
        if not np.any(early_mask):
            raise ValueError(f"No time points are less than or equal to t_cut ({t_cut}).Please choose a t_cut that includes some early time points.")

        #sequentially separate the signals, with L2 regularization on each step separately
        if alpha_1 > 0 and alpha_2 > 0:
                w_1 = nnls_l2(Phi_1[early_mask, :], y[early_mask], alpha_1)
                w_2 = nnls_l2(Phi_2, np.clip(y - Phi_1 @ w_1, 0, None), alpha_2)
        elif alpha_1 > 0 and alpha_2 == 0:
                w_1 = nnls_l2(Phi_1[early_mask, :], y[early_mask], alpha_1)
                w_2, _ = nnls(Phi_2, np.clip(y - Phi_1 @ w_1, 0, None))
        elif alpha_1 == 0 and alpha_2 > 0:
                w_1, _ = nnls(Phi_1[early_mask, :], y[early_mask])
                w_2 = nnls_l2(Phi_2, np.clip(y - Phi_1 @ w_1, 0, None), alpha_2)
        else: #no regularization
                w_1, _ = nnls(Phi_1[early_mask, :], y[early_mask])
                w_2, _ = nnls(Phi_2, np.clip(y - Phi_1 @ w_1, 0, None))
        est_1 = Phi_1 @ w_1
        est_2 = Phi_2 @ w_2
        return SeparationResult(
            tracer1_curve=est_1,
            tracer2_curve=est_2,
            coef_1 = w_1, 
            coef_2 = w_2,
            metadata={
                "w_1": w_1,
                "w_2": w_2,
                "t_cut": t_cut,
                "alpha_1": alpha_1,
                "alpha_2": alpha_2,
                "early_mask": early_mask
            }
    )