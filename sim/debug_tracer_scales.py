# scripts/debug_scale_diagnostics.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from sim.config import SimulationConfig
from sim.delay_scale_experiments import build_base_sim_objects, make_pbr_tracer
from sim.sweep import sweep_delays
from utils.run_logger import start_run, finalize_run


# ----- Paths -----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "figures" / "scale_diagnostics"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_scale_diagnostics(config: SimulationConfig,
                           delay_min: float,
                           run_description: str,
                           run_notes: str = ""):
    """
    For a fixed FDG delay (delay_min), loop over all PBR:FDG scale ratios
    in config.scale_values, and plot:

      - AIFs for PBR and FDG over time (overlay across scales)
      - Biological TACs (Ct_bio) for PBR and FDG (overlay across scales)

    Saves a single figure and logs it via start_run / finalize_run.
    """
    # ----- 1) Start logging -----
    params = {
        "diagnostic": "scale_vs_fdg",
        "delay_min": float(delay_min),
        "scale_values": config.scale_values.tolist(),
        "frame_edges": config.frame_edges.tolist(),
        "rng_seed": config.rng_seed,
    }
    run_info = start_run(params, description=run_description)

    # ----- 2) Build shared simulation objects -----
    timegrid, gamma_lib, Gamma, fdg, rng = build_base_sim_objects(config)

    # time axes
    t_dense = np.linspace(0.0, config.frame_edges[-1], 1000)
    t_frames = None  # will be filled from first simulation

    # containers for plotting
    aif_pbr_list = []
    aif_fdg = None  # same across scales
    tac_pbr_list = []
    tac_fdg_list = []

    # color mapping
    cmap = plt.get_cmap("magma")
    n_scales = len(config.scale_values)

    # ----- 3) Loop over all scale ratios -----
    for i, scale_ratio in enumerate(config.scale_values):
        pbr = make_pbr_tracer(scale_ratio)

        # AIFs: PBR inject at 0, FDG at delay_min
        Cp_pbr = pbr.aif(t_dense, delta_min=0.0)
        Cp_fdg = fdg.aif(t_dense, delta_min=delay_min)

        aif_pbr_list.append(Cp_pbr)
        if aif_fdg is None:
            aif_fdg = Cp_fdg  # FDG AIF identical across scales

        # Run a single-delay sweep to get TACs
        results = sweep_delays(
            delays=np.array([delay_min]),
            timegrid=timegrid,
            rac=pbr,
            fdg=fdg,
            gamma_lib=gamma_lib,
            Gamma=Gamma,
            rng=rng,
            use_joint=False,
        )

        sim = results[0]["sim"]
        if t_frames is None:
            t_frames = sim["t_frames"]

        Ct_pbr = sim["Ct1_bio"]   # assumes tracer 1 = PBR
        Ct_fdg = sim["Ct2_bio"]   # assumes tracer 2 = FDG

        tac_pbr_list.append(Ct_pbr)
        tac_fdg_list.append(Ct_fdg)

        print(f"[DEBUG] scale_ratio={scale_ratio:.2f}, FDG scale={fdg.scale:.3f}, "
              f"PBR scale={pbr.scale:.3f}")

    # ----- 4) Build the figure -----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_aif, ax_tac = axes

    # Plot AIFs
    for i, scale_ratio in enumerate(config.scale_values):
        color = cmap(i / max(n_scales - 1, 1))

        ax_aif.plot(
            t_dense,
            aif_pbr_list[i],
            color=color,
            alpha=0.8,
            label=f"PBR AIF (ratio={scale_ratio:.2f})" if i in (0, n_scales - 1) else None,
        )

    # FDG AIF (one curve)
    ax_aif.plot(
        t_dense,
        aif_fdg,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"FDG AIF (scale={fdg.scale:.2f}, Δ={delay_min} min)",
    )

    ax_aif.set_xlabel("Time (min)")
    ax_aif.set_ylabel("AIF (arb. units)")
    ax_aif.set_title("AIFs: PBR (varied scale) vs FDG (fixed scale)")
    ax_aif.grid(True, alpha=0.3)
    ax_aif.legend(fontsize=8)

    # Plot TACs
    for i, scale_ratio in enumerate(config.scale_values):
        color = cmap(i / max(n_scales - 1, 1))

        ax_tac.plot(
            t_frames,
            tac_pbr_list[i],
            color=color,
            alpha=0.8,
            label=f"PBR Ct (ratio={scale_ratio:.2f})" if i in (0, n_scales - 1) else None,
        )

    # You can either plot just one FDG curve (they should all be identical)
    ax_tac.plot(
        t_frames,
        tac_fdg_list[0],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="FDG Ct (fixed scale)",
    )

    ax_tac.set_xlabel("Time (min)")
    ax_tac.set_ylabel("Biological concentration")
    ax_tac.set_title("Biological TACs: PBR vs FDG")
    ax_tac.grid(True, alpha=0.3)
    ax_tac.legend(fontsize=8)

    fig.suptitle(
        f"PBR:FDG scale diagnostics at delay Δ={delay_min} min\n"
        f"scale_values={config.scale_values.tolist()}",
        fontsize=11,
    )
    fig.tight_layout()

    # ----- 5) Save figure -----
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    fname = FIG_DIR / f"scale_diagnostics_delay{int(delay_min)}_{timestamp}.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig_paths = [str(fname)]

    # ----- 6) Log the run -----
    key_params = (
        f"diagnostic=scale_vs_fdg, delay={delay_min} min, "
        f"scales={config.scale_values.tolist()}"
    )

    finalize_run(
        run_info,
        figure_filenames=fig_paths,
        key_params_str=key_params,
        notes=run_notes.strip(),
    )

    print(f"[INFO] Saved diagnostic figure to {fname}")
def main():
    # Example: use a similar framing & delays as your main simulations
    delays = np.arange(0, 50, 1)
    frame_edges = np.linspace(0.0, 120.0, 361)

    scale_values = np.array([
        0.5, 0.6, 0.7, 0.8, 0.9,
        1.0,
        1.1, 1.2, 1.3, 1.4,
        1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
    ])

    config = SimulationConfig(
        delays=delays,
        scale_values=scale_values,
        frame_edges=frame_edges,
        rng_seed=42,
    )

    RUN_DESCRIPTION = "PBR vs FDG scale diagnostic: AIFs + TACs"
    RUN_NOTES = """
    Diagnostic run to verify that FDG scale is fixed (1.0)
    and PBR amplitude varies according to scale_values.
    Plots AIFs and biological TACs over all scale ratios for a single delay.
    """

    delay_min = 20.0  # choose a representative delay (e.g., 20 min)

    plot_scale_diagnostics(
        config=config,
        delay_min=delay_min,
        run_description=RUN_DESCRIPTION,
        run_notes=RUN_NOTES,
    )


if __name__ == "__main__":
    main()
