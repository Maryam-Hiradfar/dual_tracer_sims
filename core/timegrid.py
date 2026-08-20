# core/timegrid.py
from dataclasses import dataclass
import numpy as np

@dataclass
class TimeGrid:
    frame_edges: np.ndarray          # shape (F+1,)
    internal_dt_min: float = 1/60.0  # 1 sec internal grid by default

    def __post_init__(self):
        self.frame_edges = np.asarray(self.frame_edges, dtype=float)
        if self.frame_edges.ndim != 1 or len(self.frame_edges) < 2:
            raise ValueError("frame_edges must be a one-dimensional array of length >= 2")
        if not np.isfinite(self.frame_edges).all():
            raise ValueError("frame_edges must contain only finite values")
        if not np.isfinite(self.internal_dt_min) or self.internal_dt_min <= 0:
            raise ValueError("internal_dt_min must be finite and positive")
        if np.any(np.diff(self.frame_edges) <= 0):
            raise ValueError("frame_edges must be strictly increasing")

        self.frame_mids = 0.5*(self.frame_edges[:-1] + self.frame_edges[1:])
        self.frame_durs = self.frame_edges[1:] - self.frame_edges[:-1]
        if np.any(self.frame_durs < self.internal_dt_min):
            raise ValueError(
                "every frame duration must be at least as long as internal_dt_min"
            )

        # high-res internal time vector for ODE
        self.t_internal = np.arange(
            self.frame_edges[0],
            self.frame_edges[-1] + self.internal_dt_min,
            self.internal_dt_min
        )

    @property
    def n_frames(self):
        return len(self.frame_mids)
