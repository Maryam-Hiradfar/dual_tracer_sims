# core/timegrid.py
from dataclasses import dataclass
import numpy as np

@dataclass
class TimeGrid:
    frame_edges: np.ndarray          # shape (F+1,)
    internal_dt_min: float = 1/60.0  # 1 sec internal grid by default

    def __post_init__(self):
        self.frame_edges = np.asarray(self.frame_edges, dtype=float)
        self.frame_mids = 0.5*(self.frame_edges[:-1] + self.frame_edges[1:])
        self.frame_durs = self.frame_edges[1:] - self.frame_edges[:-1]

        # high-res internal time vector for ODE
        self.t_internal = np.arange(
            self.frame_edges[0],
            self.frame_edges[-1] + self.internal_dt_min,
            self.internal_dt_min
        )

    @property
    def n_frames(self):
        return len(self.frame_mids)
    
    # core/timegrid.py (or core/utils.py)
def frame_average(t_internal, Ct_internal, frame_edges):
    Ct_frames = np.zeros(len(frame_edges)-1)
    for i in range(len(Ct_frames)):
        m = (t_internal >= frame_edges[i]) & (t_internal < frame_edges[i+1])#this is a boolean mask, it is an array of True/False values. true if t_internal is in the frame
        Ct_frames[i] = Ct_internal[m].mean() if np.any(m) else 0.0 #np.amy(m) checks if there is any True value in the mask
    return Ct_frames

