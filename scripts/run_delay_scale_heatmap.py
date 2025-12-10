# scripts/run_delay_scale_heatmap.py
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from sim.delay_scale_experiments import SimulationConfig, sweep_scale_and_delay
from utils.run_logger import start_run, finalize_run
from scripts.plot_delay_scale_heatmap import plot_nrmse_heatmap

def main():
    # ==============================
    # USER-RUN METADATA (EDIT HERE)
    # ==============================
    RUN_DESCRIPTION = "Delay × PBR-scale NRMSE heatmaps (FDG + PBR28)"
    RUN_NOTES = """
    Exploring identifiability region:
    - Delays: 0–49 min in 1-min steps
    - PBR:FDG scale ratios: 0.5–2.0
    - Fixed gamma library (dense, oracle-style)
    - Realistic non-uniform framing
    """

    KEY_PARAMS = (
        "heatmap, FDG/PBR, delays=0–49 min, "
        "scale_ratios=0.5–2.0, gamma=oracle_dense_v1"
    )

    # -------------------------------
    # USER SETTINGS FOR THIS EXPERIMENT
    # -------------------------------
    delays = np.arange(0, 50, 1)      # X-axis of heatmap
    scale_values = np.array([0.5, 0.6, 0.7, 0.8, 0.9,
                             1.0, 1.1, 1.2, 1.3, 1.4,
                             1.5, 1.6, 1.7, 1.8, 1.9, 2.0])

    frame_edges = np.array([
        0, 2, 4, 6, 8, 10,
        12, 14, 16, 18, 20,
        # ... continue realistic PET framing
        120
    ])

    config = SimulationConfig(
        delays=delays,
        scale_values=scale_values,
        frame_edges=frame_edges,
        rng_seed=42
    )
    # -------------------------------
    # LOGGING SETUP
    # -------------------------------

    # Pack params for logging (convert numpy arrays to lists)
    params = {
        "delays": delays.tolist(),
        "scale_values": scale_values.tolist(),
        "frame_edges": frame_edges.tolist(),
        "rng_seed": config.rng_seed,
        "experiment": "delay_scale_heatmap",
        "gamma_library": {
            "type": "oracle_dense_v1",
            # fill with your actual gamma settings if you want
        },
    }

    # ----- Start run logging -----
    run_info = start_run(params, description=RUN_DESCRIPTION)

    # timestamp for filenames
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")

    # -------------------------------
    # RUN THE EXPERIMENT
    # -------------------------------
    results = sweep_scale_and_delay(config)

    # -------------------------------
    # PLOT
    # -------------------------------
    plot_nrmse_heatmap(
        results["nrmse_pbr"],
        results["delays"],
        results["scales"],
        "PBR NRMSE vs Delay and PBR:FDG Scale",
        tracer_label="PBR28",
    )

    plot_nrmse_heatmap(
        results["nrmse_fdg"],
        results["delays"],
        results["scales"],
        "FDG NRMSE vs Delay and PBR:FDG Scale",
        tracer_label="FDG",
    )

    plt.show()


if __name__ == "__main__":
    main()
