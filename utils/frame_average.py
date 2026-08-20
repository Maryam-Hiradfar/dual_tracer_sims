# utils/frame_average.py
import numpy as np


def frame_average(t_internal, Ct_internal, frame_edges):
    """Average a fine-grid TAC over frames using piecewise-linear integration."""
    t_internal = np.asarray(t_internal, dtype=float)
    Ct_internal = np.asarray(Ct_internal, dtype=float)
    frame_edges = np.asarray(frame_edges, dtype=float)

    if t_internal.ndim != 1 or Ct_internal.ndim != 1:
        raise ValueError("t_internal and Ct_internal must be one-dimensional")
    if t_internal.shape != Ct_internal.shape:
        raise ValueError("t_internal and Ct_internal must have the same shape")
    if len(t_internal) < 2 or np.any(np.diff(t_internal) <= 0):
        raise ValueError("t_internal must contain at least two increasing values")
    if frame_edges.ndim != 1 or len(frame_edges) < 2:
        raise ValueError("frame_edges must be a one-dimensional array of length >= 2")
    if np.any(np.diff(frame_edges) <= 0):
        raise ValueError("frame_edges must be strictly increasing")
    if frame_edges[0] < t_internal[0] or frame_edges[-1] > t_internal[-1]:
        raise ValueError("the internal time grid must cover every frame edge")

    Ct_frames = np.empty(len(frame_edges) - 1, dtype=float)
    for i, (start, stop) in enumerate(zip(frame_edges[:-1], frame_edges[1:])):
        interior = (t_internal > start) & (t_internal < stop)
        frame_times = np.concatenate(([start], t_internal[interior], [stop]))
        frame_values = np.interp(frame_times, t_internal, Ct_internal)
        Ct_frames[i] = np.trapezoid(frame_values, frame_times) / (stop - start)
    return Ct_frames
