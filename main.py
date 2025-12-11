# main.py

import numpy as np
import matplotlib.pyplot as plt

from core.timegrid import TimeGrid
from tracers.rac import RacloprideTracer
from tracers.fdg import FDGTracer
from tracers.pbr28 import PBR28Tracer

from kinetics.basis_gamma import build_gamma_library
from sim.sweep import sweep_delays
from datetime import datetime
import os
from utils.run_logger import start_run, finalize_run, get_figure_path

FIG_DIR = "/Users/maryamhiradfar/desktop/catana_lab_local/dual_tracer/dualtracer/figures/PBR_scale_1.2_fdg_scale_1"

# Create directory if it doesn't exist
os.makedirs(FIG_DIR, exist_ok=True)




def main():
    # ==============================
    # USER-RUN SETTINGS (EDIT HERE)
    # ==============================

    RUN_DESCRIPTION = "FDG/PBR delay sweep (#2) with updated Feng params"
    RUN_NOTES = """
    Testing new gamma basis library (version 3)
    - FDG scale = 1.0
    - PBR scale = 1.0
    - Delays: 0–50 min (1-min steps)
    - Frame edges:frame_durations_min = np.array(
        [2/60.0] * 20    # 20 × 2 s  = 40 s
        + [5/60.0] * 12    # 12 × 5 s  = 60 s
        + [10/60.0] * 12   # 12 × 10 s = 120 s
        + [20/60.0] * 12   # 12 × 20 s = 240 s
        + [30/60.0] * 8    # 8 × 30 s  = 240 s
        + [1.0]     * 6    # 6 × 60 s  = 360 s
        + [2.0]     * 5    # 5 × 120 s = 600 s
        + [5.0]     * 8    # 8 × 300 s = 2400 s
        + [10.0]    * 4    # 4 × 600 s = 2400 s
    )
    - Gamma basis: n_t0 = 70, n_tau = 80
    """
    GAMMA_N_T0 = 70
    GAMMA_N_TAU = 80
    key_params = "FDG/PBR, delay sweep, Feng AIF, frame_edges=0–120, realistic_nonuniform, PBR scale=1.0, FDG scale=1.0, gamma=(n_t0=70, n_tau=80)"
    

    #optional overrides for default parameters
    FDG_OVERRIDES = {
        # "A1": 1854.66,
        # "A2": 8.168,
        # "A3": 2.731,
        # "lam1": 20.031,
        # "lam2": 0.355,
        # "lam3": 0.0178,
        # "tau": 0.27
    }
    PBR_OVERRIDES = {
        # "A1": 22.0,
        # "A2": 3.2,
        # "A3": 0.25,
        # "lam1": 3.1,
        # "lam2": 0.23,
        # "lam3": 0.015
    }
    # ==============================
    # Frame durations in MINUTES
    frame_durations_min = np.array(
        [2/60.0] * 20    # 20 × 2 s  = 40 s
        + [5/60.0] * 12    # 12 × 5 s  = 60 s
        + [10/60.0] * 12   # 12 × 10 s = 120 s
        + [20/60.0] * 12   # 12 × 20 s = 240 s
        + [30/60.0] * 8    # 8 × 30 s  = 240 s
        + [1.0]     * 6    # 6 × 60 s  = 360 s
        + [2.0]     * 5    # 5 × 120 s = 600 s
        + [5.0]     * 8    # 8 × 300 s = 2400 s
        + [10.0]    * 4    # 4 × 600 s = 2400 s
    )
    FRAME_DURATIONS_MIN  = frame_durations_min
    FRAME_EDGES = np.concatenate([[0.0], np.cumsum(frame_durations_min)])

    DELAYS = np.arange(0, 50, 1)  # Δ = 0..15 min this is for the delay sweep
    FRAME_EDGES = np.linspace(0.0, 120.0, 361)  # [0, 2, 4, ..., 120], this is for section "Define dynamic frame schedule"
        # Example: 0–120 min in 2-min frames → 61 edges, 60 frames
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")

    # ----- 2) Define tracers -----
    # Half-lives: 11C ≈ 20.4 min, 18F ≈ 109.8 min
    pbr = PBR28Tracer(
        name="PBR28",
        half_life_min=20.4, 
        scale = 1.0,
        pbr28_params = (PBR_OVERRIDES if PBR_OVERRIDES else None),
        )
    fdg = FDGTracer(
        name="FDG", 
        half_life_min=109.8,
        scale = 1.0, 
        feng_params = (FDG_OVERRIDES if FDG_OVERRIDES else None),
        )
    
    #===============================
    # ------- Run logging setup --------#
    #===============================


    params = {
    "run_description" : RUN_DESCRIPTION,
    "tracers" : ["FDG", "PBR28"], 
    "fdg":{
        "name" : "FDG",
        "half_life_min" : fdg.half_life_min,
        "scale" : fdg.scale,
        "feng_params" : fdg.feng_params,
    },  
    "pbr":{
        "name" : "PBR28",
        "half_life_min" : pbr.half_life_min,
        "scale" : pbr.scale,
        "pbr28_params" : pbr.pbr28_params,
    },   
    "delays" : DELAYS,
    "frame_edges" : FRAME_EDGES, 
    "gamma_basis": {
            "n_t0": GAMMA_N_T0,
            "n_tau": GAMMA_N_TAU,
        },
    }
    run_info = start_run(params, description="Delay sweep for PBR28 and FDG with PBR scale 1.2 and FDG scale 1.0")
    #=======================================================
    # 3) Build time grid, gamma basis, and run delay sweep
    # =======================================================
    # ----- 1) Define dynamic frame schedule -----
    timegrid = TimeGrid(frame_edges=FRAME_EDGES, internal_dt_min=1/60.0)  # 1 sec internal
    gamma_lib, Gamma = build_gamma_library(
        timegrid.frame_mids, 
        n_t0=GAMMA_N_T0,
        n_tau=GAMMA_N_TAU,
        )

    # ----- 4) Prepare delay sweep -----
    rng = np.random.default_rng(42)

    results = sweep_delays(
        delays=DELAYS,
        timegrid=timegrid,
        rac=pbr,
        fdg=fdg,
        gamma_lib=gamma_lib,
        Gamma=Gamma,
        rng=rng,
        use_joint=False,    # we only have sequential_nnls for now
    )

    # results is a list of dicts; convert to arrays for plotting
    nrmse_pbr = np.array([r["nrmse_pbr_bio"] for r in results])
    nrmse_fdg = np.array([r["nrmse_fdg_bio"] for r in results])

    #==================================
    # 4) Generate and save figures
    #==================================

    fig_paths: list[str] = []

    # ----- figure 1:  Plot NRMSE vs Δ -----
    plt.figure(figsize=(7, 5))
    plt.plot(DELAYS, nrmse_pbr, "-o", label="PBR NRMSE (bio)")
    plt.plot(DELAYS, nrmse_fdg, "-o", label="FDG NRMSE (bio)")
    plt.xlabel("FDG injection delay Δ (min)")
    plt.ylabel("NRMSE")
    plt.title("Unmixing error vs injection delay")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0,0.6)
    plt.tight_layout()
    fig1_path = get_figure_path(run_info, f"nrmse_vs_delay_{timestamp}.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    fig_paths.append(fig1_path)

    # ===== FIGURE 2: Grid of true vs estimated biological TACs across delays, split into multiple files =====
   
    n_results = len(results)
    n_cols = 4
    n_rows_per_fig = 3    # adjust here
    plots_per_fig = n_cols * n_rows_per_fig

    n_figs = int(np.ceil(n_results / plots_per_fig))

    for f in range(n_figs):
        start = f * plots_per_fig
        end = min(start + plots_per_fig, n_results)
        chunk = results[start:end]

        # number of rows for this figure
        n_rows = int(np.ceil(len(chunk) / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(18, 4 * n_rows),
            sharex=True, sharey=True
        )

        # Keep axes 2D for consistency
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for idx, r in enumerate(chunk):
            i, j = divmod(idx, n_cols)
            ax = axes[i, j]

            D = r["Delta"]
            sim = r["sim"]

            t = sim["t_frames"]
            Ct1_true, Ct1_est = sim["Ct1_bio"], sim["Ct1_est_bio"]
            Ct2_true, Ct2_est = sim["Ct2_bio"], sim["Ct2_est_bio"]

            ax.plot(t, Ct1_true, lw=2, label="PBR true")
            ax.plot(t, Ct1_est, "--", lw=1.2, label="PBR est")
            ax.plot(t, Ct2_true, lw=2, label="FDG true")
            ax.plot(t, Ct2_est, "--", lw=1.2, label="FDG est")

            ax.set_title(f"Δ = {D} min")
            ax.grid(True, alpha=0.2)

        # turn off unused axes in last row
        total_slots = n_rows * n_cols
        for k in range(len(chunk), total_slots):
            i, j = divmod(k, n_cols)
            axes[i, j].axis("off")

        # global legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=12)

        fig.text(0.5, 0.04, "Time (min)", ha="center", fontsize=14)
        fig.text(0.04, 0.5, "Biological concentration", va="center",
                rotation="vertical", fontsize=14)
        fig2_path = get_figure_path(run_info, f"tac_grid_part{f+1}_{timestamp}.png")
        plt.tight_layout(rect=[0.03, 0.05, 1, 0.95])
        plt.savefig(fig2_path, dpi=300)
        plt.close()
        fig_paths.append(fig2_path)




    # ===== FIGURE 3: FDG biological TACs aligned by time since FDG injection =====
    plt.figure(figsize=(8, 5))

    for r in results:
        D = r["Delta"]
        sim = r["sim"]

        t = sim["t_frames"]
        Ct2_true = sim["Ct2_bio"]

        tau = np.maximum(t - D, 0.0)
        mask = t >= D

        plt.plot(tau[mask], Ct2_true[mask], label=f"Δ={D} min")

    plt.xlabel("Time since FDG injection τ (min)")
    plt.ylabel("FDG biological TAC")
    plt.title("FDG biological TACs aligned by biological time")
    plt.xlim(0, 60)
    plt.grid(True, alpha=0.3)

    plt.legend(
        bbox_to_anchor=(1.02, 1),      # place legend outside right
        loc="upper left",
        borderaxespad=0
    )

    plt.tight_layout()
    fig3_path = get_figure_path(run_info, f"fdg_aligned_true_{timestamp}.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    fig_paths.append(fig3_path)


    # ===== FIGURE 4: Estimated FDG biological TACs aligned by time since FDG injection =====
    plt.figure(figsize=(8, 5))

    for r in results:
        D = r["Delta"]
        sim = r["sim"]

        t = sim["t_frames"]
        Ct2_est = sim["Ct2_est_bio"]

        tau = np.maximum(t - D, 0.0)
        mask = t >= D

        plt.plot(tau[mask], Ct2_est[mask], "--", label=f"Δ={D} min")

    plt.xlabel("Time since FDG injection τ (min)")
    plt.ylabel("Estimated FDG biological TAC")
    plt.title("Estimated FDG TACs aligned by biological time")
    plt.xlim(0, 60)
    plt.grid(True, alpha=0.3)

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0
    )

    plt.tight_layout()
    fig4_path = get_figure_path(run_info, f"fdg_aligned_estimated_{timestamp}.png")
    plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close()
    fig_paths.append(fig4_path)

    # ------- Figure 4: gamma library----------#
    plt.figure(figsize=(8,5))
    t_basis_plot = np.linspace(t[0], t[-1], 360)

    for g in gamma_lib:
        plt.plot(t_basis_plot, g, alpha = 0.3)
    plt.xlabel("Time (min)")
    plt.ylabel("Amplitude")
    plt.title("Gamma basis library (all functions)")
    plt.grid(True, alpha = 0.3)

    plt.tight_layout()
    fig5_path = get_figure_path(run_info, f"gamma_library_{timestamp}.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    fig_paths.append(fig5_path)
    #==================================
    # ------- Finalize run logging ----------#
    #==================================
    finalize_run(
        run_info,
        figure_filenames=fig_paths,
        key_params_str=key_params,
        notes= RUN_NOTES.strip(),
    )
    






if __name__ == "__main__":
    main()
