import numpy as np

def default_frame_edges():
    # frame_durations_min = np.array(
    #     [2/60.0] * 20    # 20 × 2 s  = 40 s
    #     + [5/60.0] * 12    # 12 × 5 s  = 60 s
    #     + [10/60.0] * 12   # 12 × 10 s = 120 s
    #     + [20/60.0] * 12   # 12 × 20 s = 240 s
    #     + [30/60.0] * 8    # 8 × 30 s  = 240 s
    #     + [1.0]     * 6    # 6 × 60 s  = 360 s
    #     + [2.0]     * 5    # 5 × 120 s = 600 s
    #     + [5.0]     * 8    # 8 × 300 s = 2400 s
    #     + [10.0]    * 4    # 4 × 600 s = 2400 s
    # )
    frame_durations_min = np.array(
        [5/60.0] * 14    # 14 × 5 s  = 70 s
        + [30/60.0] * 3    # 3 × 30 s  = 90 s
        + [60/60.0] * 5   # 5 × 60 s = 300 s
        + [120/60.0] * 40   # 40 × 120 s = 4800 s
    )

    return np.concatenate(([0.0], np.cumsum(frame_durations_min)))
def uniform_frame_edges(tmax:float = 120.0, dt: float = 2.0) -> np.ndarray:
    return np.linspace(0.0, tmax, int(tmax/dt) + 1)

