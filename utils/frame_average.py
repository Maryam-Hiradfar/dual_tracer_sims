# utils/frame_average.py
import numpy as np

def frame_average(t_internal, Ct_internal, frame_edges):
    Ct_frames = np.zeros(len(frame_edges)-1)
    for i in range(len(Ct_frames)):
        m = (t_internal >= frame_edges[i]) & (t_internal < frame_edges[i+1])
        Ct_frames[i] = Ct_internal[m].mean() if np.any(m) else 0.0
    return Ct_frames
