# # scripts/run_delay_scale_heatmap.py
## Old code, needs to be updated to work with refactored code (06/12/2026) - Maryam Hiradfar
# import os
# from pathlib import Path
# from datetime import datetime

# import numpy as np
# import matplotlib.pyplot as plt

# from sim.delay_scale_experiments import SimulationConfig, sweep_scale_and_delay
# from utils.run_logger import start_run, finalize_run
# from scripts.plot_delay_scale_heatmap import plot_nrmse_heatmap

# def main():
#     # ==============================
#     # USER-RUN METADATA (EDIT HERE)
#     # ==============================
#     RUN_DESCRIPTION = "Delay × PBR-scale NRMSE heatmaps (FDG + PBR28)"
#     RUN_NOTES = """
#     Exploring identifiability region:
#     - Delays: 0–49 min in 1-min steps
#     - PBR:FDG scale ratios: 0.5–2.0
#     - Fixed gamma library (nt0 =50, ntau = 50)
#     - Realistic non-uniform framing
#     - Frame edges:frame_durations_min = np.array(
#         [2/60.0] * 20    # 20 × 2 s  = 40 s
#         + [5/60.0] * 12    # 12 × 5 s  = 60 s
#         + [10/60.0] * 12   # 12 × 10 s = 120 s
#         + [20/60.0] * 12   # 12 × 20 s = 240 s
#         + [30/60.0] * 8    # 8 × 30 s  = 240 s
#         + [1.0]     * 6    # 6 × 60 s  = 360 s
#         + [2.0]     * 5    # 5 × 120 s = 600 s
#         + [5.0]     * 8    # 8 × 300 s = 2400 s
#         + [10.0]    * 4    # 4 × 600 s = 2400 s
#     )
#     """

#     KEY_PARAMS = (
#         "heatmap, FDG/PBR, delays=0–49 min, "
#         "scale_ratios=0.5–2.0, gamma=(nt0 = 50, ntau = 50)"
#         ", framing=realistic_nonuniform"
#     )

#     # -------------------------------
#     # USER SETTINGS FOR THIS EXPERIMENT
#     # -------------------------------
#     FIG_DIR = Path("/Users/maryamhiradfar/Documents/GitHub/dual_tracer_sims/figures")
#     FIG_DIR.mkdir(parents=True, exist_ok=True) 
#     delays = np.arange(0, 50, 1)      # X-axis of heatmap
#     scale_values = np.array([0.5, 0.6, 0.7, 0.8, 0.9,
#                              1.0, 1.1, 1.2, 1.3, 1.4,
#                              1.5, 1.6, 1.7, 1.8, 1.9, 2.0])

#     # Frame durations in MINUTES
#     frame_durations_min = np.array(
#         [2/60.0] * 20    # 20 × 2 s  = 40 s
#         + [5/60.0] * 12    # 12 × 5 s  = 60 s
#         + [10/60.0] * 12   # 12 × 10 s = 120 s
#         + [20/60.0] * 12   # 12 × 20 s = 240 s
#         + [30/60.0] * 8    # 8 × 30 s  = 240 s
#         + [1.0]     * 6    # 6 × 60 s  = 360 s
#         + [2.0]     * 5    # 5 × 120 s = 600 s
#         + [5.0]     * 8    # 8 × 300 s = 2400 s
#         + [10.0]    * 4    # 4 × 600 s = 2400 s
#     )
    
#     frame_edges = np.concatenate([[0.0], np.cumsum(frame_durations_min)])

#     config = SimulationConfig(
#         delays=delays,
#         scale_values=scale_values,
#         frame_edges=frame_edges,
#         rng_seed=42
#     )
#     # -------------------------------
#     # LOGGING SETUP
#     # -------------------------------

#     # Pack params for logging (convert numpy arrays to lists)
#     params = {
#         "delays": delays.tolist(),
#         "scale_values": scale_values.tolist(),
#         "frame_edges": frame_edges.tolist(),
#         "rng_seed": config.rng_seed,
#         "experiment": "delay_scale_heatmap",
#         "gamma_library": {
#             "type": "oracle_dense_v1",
#             # fill with your actual gamma settings if you want
#         },
#     }

#     # ----- Start run logging -----
#     run_info = start_run(params, description=RUN_DESCRIPTION)

#     # timestamp for filenames
#     timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")

#     # -------------------------------
#     # RUN THE EXPERIMENT
#     # -------------------------------
#     results = sweep_scale_and_delay(config)

#     nrmse_pbr = results["nrmse_pbr"]   # shape: (n_scales, n_delays)
#     nrmse_fdg = results["nrmse_fdg"]
#     delays = results["delays"]
#     scales = results["scales"]

#     # -------------------------------
#     # PLOT AND SAVE HEATMAPS
#     # -------------------------------
#     fig_paths = []

#     # PBR heatmap
#     fig_pbr = plot_nrmse_heatmap(
#         nrmse_pbr,
#         delays,
#         scales,
#         title="PBR NRMSE vs delay and PBR:FDG scale",
#         tracer_label="PBR28",
#         vmin=0, 
#         vmax=0.1,
#     )
#     fname_pbr = FIG_DIR / f"heatmap_pbr_nrmse_{timestamp}.png"
#     fig_pbr.savefig(fname_pbr, dpi=300, bbox_inches="tight")
#     plt.close(fig_pbr)
#     fig_paths.append(str(fname_pbr))

#     # FDG heatmap
#     fig_fdg = plot_nrmse_heatmap(
#         nrmse_fdg,
#         delays,
#         scales,
#         title="FDG NRMSE vs delay and PBR:FDG scale",
#         tracer_label="FDG",
#         vmin=0,
#         vmax=0.1,
#     )
#     fname_fdg = FIG_DIR / f"heatmap_fdg_nrmse_{timestamp}.png"
#     fig_fdg.savefig(fname_fdg, dpi=300, bbox_inches="tight")
#     plt.close(fig_fdg)
#     fig_paths.append(str(fname_fdg))

#     # (Optional) you can also show them interactively when running by hand:
#     # plt.show()

#     # -------------------------------
#     # FINALIZE LOGGING
#     # -------------------------------
#         # ===== 4) Finalize run logging =====
#     finalize_run(
#         run_info,
#         figure_filenames=fig_paths,
#         key_params_str=KEY_PARAMS,
#         notes=RUN_NOTES.strip(),
#     )

# if __name__ == "__main__":
#     main()
