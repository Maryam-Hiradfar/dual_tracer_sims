# # scripts/run_gamma_3d_sweep.py
# Old code, needs to get updated to become compatible with the refactoring (06/12/2026) -Maryam Hiradfar

# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# from datetime import datetime

# from sim.config import SimulationConfig
# from sim.delay_scale_experiments import sweep_scale_and_delay

# # If you have your nice heatmap helper already:
# # from scripts.plot_heatmaps import plot_nrmse_heatmap

# # ---------- Paths ----------
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# OUT_DIR = Path("/Users/maryamhiradfar/Documents/GitHub/dual_tracer_sims/figures")
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# def run_gamma_3d_sweep(
#     use_global_vrange: bool = True,
#     vmin: float | None = None,
#     vmax: float | None = None,
#     vrange_percentiles: tuple[int, int] = (5, 95),
# ):
#     """
#     3-parameter sweep over:
#       - delays
#       - scale_values (PBR:FDG ratios)
#       - gamma library sizes (n_t0, n_tau)

#     vmin/vmax options:
#       - If use_global_vrange=True and (vmin or vmax is None):
#           compute global vmin/vmax from all PBR NRMSE values using vrange_percentiles.
#       - If use_global_vrange=False and vmin/vmax are provided:
#           use those fixed values everywhere.
#       - If neither is set (use_global_vrange=False and vmin/vmax=None):
#           each heatmap gets its own auto scaling (not recommended for cross-comparison).

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
#     # ----- 1) Define base delay/scale/grid -----
#     delays = np.arange(0, 50, 1)
#     scale_values = np.array([
#         0.5, 0.6, 0.7, 0.8, 0.9,
#         1.0,
#         1.1, 1.2, 1.3, 1.4,
#         1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
#     ])

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
#     n_t0_values  = [50, 70, 80, 90, 100]
#     n_tau_values = [50, 70, 80, 90, 100]

#     results_pbr = {}

#     # ------------------------------------------------------------------
#     # 2) First pass: run sweeps and store NRMSE matrices (no plotting)
#     # ------------------------------------------------------------------
#     for n_t0 in n_t0_values:
#         for n_tau in n_tau_values:
#             print(f"[INFO] Running sweep for n_t0={n_t0}, n_tau={n_tau} ...")

#             config = SimulationConfig(
#                 delays=delays,
#                 scale_values=scale_values,
#                 frame_edges=frame_edges,
#                 rng_seed=42,
#                 fdg_scale=1.0,
#                 n_t0=n_t0,
#                 n_tau=n_tau,
#             )

#             sweep_out = sweep_scale_and_delay(config)
#             nrmse_pbr = sweep_out["nrmse_pbr"]  # (n_scales, n_delays)

#             results_pbr[(n_t0, n_tau)] = nrmse_pbr

#     # ------------------------------------------------------------------
#     # 3) Decide on vmin/vmax for plotting
#     # ------------------------------------------------------------------
#     if use_global_vrange:
#         # If user didn't explicitly give vmin/vmax, compute from data
#         if vmin is None or vmax is None:
#             all_vals = np.concatenate([m.ravel() for m in results_pbr.values()])
#             p_lo, p_hi = vrange_percentiles
#             auto_vmin = np.percentile(all_vals, p_lo)
#             auto_vmax = np.percentile(all_vals, p_hi)
#             vmin = auto_vmin if vmin is None else vmin
#             vmax = auto_vmax if vmax is None else vmax
#             print(f"[INFO] Global vmin/vmax set to [{vmin:.3f}, {vmax:.3f}] "
#                   f"from percentiles {vrange_percentiles}")
#         else:
#             print(f"[INFO] Using user-provided global vmin/vmax = [{vmin}, {vmax}]")
#     else:
#         if vmin is not None or vmax is not None:
#             print(f"[INFO] Using fixed vmin/vmax = [{vmin}, {vmax}] for all plots.")
#         else:
#             print("[WARN] No global vmin/vmax; each heatmap will auto-scale independently.")

#     # ------------------------------------------------------------------
#     # 4) Second pass: make individual heatmaps for each (n_t0, n_tau)
#     # ------------------------------------------------------------------
#     for (n_t0, n_tau), nrmse_pbr in results_pbr.items():
#         fig = plt.figure(figsize=(3, 2.5))
#         ax = fig.add_subplot(111)

#         im = ax.imshow(
#             nrmse_pbr,
#             aspect="auto",
#             origin="lower",
#             extent=[delays[0], delays[-1], scale_values[0], scale_values[-1]],
#             cmap="magma",
#             vmin=vmin,
#             vmax=vmax,
#         )
#         ax.set_title(f"PBR NRMSE\nn_t0={n_t0}, n_tau={n_tau}", fontsize=8)
#         ax.set_xlabel("Δ (min)", fontsize=7)
#         ax.set_ylabel("scale", fontsize=7)
#         ax.tick_params(labelsize=6)

#         cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#         cbar.ax.tick_params(labelsize=6)

#         fig.tight_layout()

#         timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
#         fname = OUT_DIR / f"pbr_heatmap_nt0{n_t0}_ntau{n_tau}_{timestamp}.png"
#         fig.savefig(fname, dpi=200)
#         plt.close(fig)

#     # ------------------------------------------------------------------
#     # 5) Make combined panel with same vmin/vmax (if any)
#     # ------------------------------------------------------------------
#     n_rows = len(n_t0_values)
#     n_cols = len(n_tau_values)

#     fig, axes = plt.subplots(
#         n_rows, n_cols,
#         figsize=(2.2 * n_cols, 2.0 * n_rows),
#         sharex=True,
#         sharey=True,
#     )

#     for i, n_t0 in enumerate(n_t0_values):
#         for j, n_tau in enumerate(n_tau_values):
#             ax = axes[i, j] if n_rows > 1 else axes[j]
#             nrmse_pbr = results_pbr[(n_t0, n_tau)]

#             im = ax.imshow(
#                 nrmse_pbr,
#                 aspect="auto",
#                 origin="lower",
#                 extent=[delays[0], delays[-1], scale_values[0], scale_values[-1]],
#                 cmap="magma",
#                 vmin=vmin,
#                 vmax=vmax,
#             )

#             if i == 0:
#                 ax.set_title(f"n_tau={n_tau}", fontsize=8)
#             if j == 0:
#                 ax.set_ylabel(f"n_t0={n_t0}\nscale", fontsize=8)
#             else:
#                 ax.set_ylabel("")

#             if i == n_rows - 1:
#                 ax.set_xlabel("Δ (min)", fontsize=8)
#             else:
#                 ax.set_xlabel("")

#             ax.tick_params(labelsize=6)

#     # Add one colorbar to the right of the entire grid
#     # [left, bottom, width, height] in figure coordinates (0–1)
#     cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.75])
#     cbar = fig.colorbar(im, cax=cbar_ax)
#     cbar.ax.tick_params(labelsize=8)
#     cbar.set_label("PBR NRMSE (bio)", fontsize=9)


#     fig.suptitle(
#         "PBR NRMSE vs Δ and scale\nacross gamma library sizes (n_t0, n_tau)",
#         fontsize=10,
#     )
#     fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])

#     grid_fname = OUT_DIR / "pbr_nrmse_gamma_3d_panel.png"
#     fig.savefig(grid_fname, dpi=300)
#     plt.close(fig)

#     print(f"[INFO] Saved combined panel to {grid_fname}")

# if __name__ == "__main__":
#     run_gamma_3d_sweep(
#         use_global_vrange=False,
#         vmin=0.0,
#         vmax=0.1,
#     )
