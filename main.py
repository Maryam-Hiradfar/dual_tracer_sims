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

FIG_DIR = "/Users/maryamhiradfar/desktop/catana_lab_local/dual_tracer/dualtracer/figures/PBR_scale_1.2_fdg_scale_1"

# Create directory if it doesn't exist
os.makedirs(FIG_DIR, exist_ok=True)




def main():
    
    pbr_scale = 1.2
    fdg_scale = 1.0
    # ----- 1) Define dynamic frame schedule -----
    # Example: 0–120 min in 2-min frames → 61 edges, 60 frames
    frame_edges = np.linspace(0.0, 120.0, 361)  # [0, 2, 4, ..., 120]
    timegrid = TimeGrid(frame_edges=frame_edges, internal_dt_min=1/60.0)  # 1 sec internal

    # ----- 2) Define tracers -----
    # Half-lives: 11C ≈ 20.4 min, 18F ≈ 109.8 min
    pbr = PBR28Tracer(name="PBR28", half_life_min=20.4, scale = pbr_scale)

    feng_params = dict(
        A1=1854.66, A2=8.168, A3=2.731,
        lam1=20.031, lam2=0.355, lam3=0.0178,
        tau=0.27
    )
    fdg = FDGTracer(name="FDG", half_life_min=109.8, feng_params=feng_params, scale = fdg_scale)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")

    # ----- 3) Build gamma basis library on FRAME times -----
    # Important: Gamma's time axis must match Ct_bio (which lives on frame_mids)
    gamma_lib, Gamma = build_gamma_library(timegrid.frame_mids)

    # ----- 4) Prepare delay sweep -----
    Delays = np.arange(0, 50, 1)  # Δ = 0..15 min
    rng = np.random.default_rng(42)

    results = sweep_delays(
        delays=Delays,
        timegrid=timegrid,
        rac=pbr,
        fdg=fdg,
        gamma_lib=gamma_lib,
        Gamma=Gamma,
        rng=rng,
        use_joint=False,    # we only have sequential_nnls for now
    )

    # results is a list of dicts; convert to arrays for plotting
    nrmse_rac = np.array([r["nrmse_pbr_bio"] for r in results])
    nrmse_fdg = np.array([r["nrmse_fdg_bio"] for r in results])

    # ----- 5) Plot NRMSE vs Δ -----
    plt.figure(figsize=(7, 5))
    plt.plot(Delays, nrmse_rac, "-o", label="PBR NRMSE (bio)")
    plt.plot(Delays, nrmse_fdg, "-o", label="FDG NRMSE (bio)")
    plt.xlabel("FDG injection delay Δ (min)")
    plt.ylabel("NRMSE")
    plt.title("Unmixing error vs injection delay")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0,0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"nrmse_vs_delay_{timestamp}.png"), dpi=300)
    # plt.show()   # optional
    plt.close()


        # ===== FIGURE 2: Grid of true vs estimated biological TACs across delays =====
    # ===== FIGURE 2: Split tiled grid into multiple files =====
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

        fname = f"tac_grid_part{f+1}_{timestamp}.png"
        plt.tight_layout(rect=[0.03, 0.05, 1, 0.95])
        plt.savefig(os.path.join(FIG_DIR, fname), dpi=300)
        plt.close()




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
    plt.savefig(os.path.join(FIG_DIR, f"fdg_aligned_true_{timestamp}.png"), dpi=300, bbox_inches="tight")
    plt.close()


        # OPTIONAL: overlay estimated FDG TACs in aligned biological time
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
    plt.savefig(os.path.join(FIG_DIR, f"fdg_aligned_estimated_{timestamp}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ------- Figure 5: gamma library----------#
    plt.figure(figsize=(8,5))
    t_basis_plot = np.linspace(t[0], t[-1], 360)

    for g in gamma_lib:
        plt.plot(t_basis_plot, g, alpha = 0.3)
    plt.xlabel("Time (min)")
    plt.ylabel("Amplitude")
    plt.title("Gamma basis library (all functions)")
    plt.grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"gamma_library_{timestamp}.png"), dpi = 300)
    plt.close()






if __name__ == "__main__":
    main()
