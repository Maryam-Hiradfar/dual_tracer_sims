"""Local-only visual QC for empirical Bq/mL dual-tracer simulation.

No participant data are uploaded. Outputs are local PNGs; do not commit them.
Run from repository root: python -m sim.validate_bqml --fdg-csv ... --pbr-csv ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.bqml_calibration import (
    HALF_LIVES_MIN, MeasuredIDIF, noisy_bqml_from_counts, simulate_mixture
)


def plot_validation(
    fdg_csv: Path,
    pbr_csv: Path,
    output_dir: Path,
    *,
    duration_min: float = 60.0,
    offsets_min: tuple[float, ...] = (0.0, 5.0, 10.0),
    signal_source: str = "tissue-approx",
    effective_sensitivity_cps_per_bq: float = 0.001,
    seed: int = 48,
    fdg_kinetics: dict[str, float] | None = None,
    fdg_vb: float = 0.0,
    fdg_delay_sec: float = 0.0,
) -> None:
    """Produce quantitative checks and visualizations, not a realism verdict."""
    fdg = MeasuredIDIF.from_csv(fdg_csv, "FDG")
    pbr = MeasuredIDIF.from_csv(pbr_csv, "PBR28")
    if duration_min <= 0 or duration_min > pbr.coverage_min:
        raise ValueError("duration must fit PBR28 IDIF coverage")
    if not offsets_min or any(x < 0 for x in offsets_min):
        raise ValueError("offsets must be nonempty and nonnegative")
    if duration_min > fdg.coverage_min + min(offsets_min):
        raise ValueError("duration exceeds FDG IDIF coverage at minimum offset")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot actual measured frame means as frame-width bars, without pretending
    # the empirical acquisition provided an instantaneous concentration curve.
    fig, axs = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    for ax, idif, name in zip(axs, (pbr, fdg), ("PBR28", "FDG")):
        t = idif.mid_min
        widths = idif.end_min - idif.start_min
        ax.bar(t, idif.bqml, width=widths, alpha=0.5, align="center",
               label="Measured decay-corrected frame mean")
        ax.plot(t, idif.bqml, "o", ms=2)
        ax.set(title=f"{name} aortic whole-blood IDIF (reference {idif.reference_dose_mci:g} mCi)",
               ylabel="Activity (Bq/mL)", xlabel="Minutes from reference scan start")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(output_dir / "01_measured_idifs.png", dpi=170)
    plt.close(fig)

    edges = np.r_[pbr.start_min[pbr.start_min < duration_min], duration_min]
    if np.any(np.diff(edges) <= 0):
        raise ValueError("invalid output frame edges")
    mid = (edges[:-1] + edges[1:]) / 2
    durations = np.diff(edges)
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(len(offsets_min), 1, figsize=(12, 3.6 * len(offsets_min)),
                             squeeze=False, constrained_layout=True)
    fig_noise, noise_axes = plt.subplots(len(offsets_min), 1,
                                       figsize=(12, 3.6 * len(offsets_min)),
                                       squeeze=False, constrained_layout=True)
    for j, offset in enumerate(offsets_min):
        sample = simulate_mixture(fdg, pbr, edges, fdg_injection_min=offset,
                                  signal_source=signal_source,
                                  fdg_kinetics=fdg_kinetics,
                                  fdg_vb=fdg_vb, fdg_delay_sec=fdg_delay_sec)
        fdg_phys, pbr_phys = sample["Y_phys_tavg_n0_d1"]
        clean = sample["X_clean_tavg_n0_d1"][0]
        np.testing.assert_allclose(clean, fdg_phys + pbr_phys, atol=1e-8)
        pre = edges[1:] <= offset
        if np.any(pre):
            np.testing.assert_allclose(fdg_phys[pre], 0, atol=1e-8)
        noisy, expected, counts = noisy_bqml_from_counts(
            clean, durations, effective_sensitivity_cps_per_bq=
            effective_sensitivity_cps_per_bq, rng=rng
        )
        ax = axes[j, 0]
        ax.plot(mid, fdg_phys, label="FDG physical", lw=1.5)
        ax.plot(mid, pbr_phys, label="PBR28 physical", lw=1.5)
        ax.plot(mid, clean, label="Physical mixture (sum)", lw=2, color="black")
        ax.axvline(offset, color="gray", ls="--", lw=1,
                   label="FDG injection" if j == 0 else None)
        ax.set(title=f"{signal_source}; FDG injection offset {offset:g} min",
               ylabel="Frame-mean activity (Bq/mL)", xlabel="Minutes from PBR28 scan start")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

        ax = noise_axes[j, 0]
        ax.plot(mid, clean, label="Clean mixture", lw=1.8)
        ax.plot(mid, noisy, label="Noisy mixture (placeholder noise model)", alpha=0.8)
        ax.set(title=f"FDG offset {offset:g} min; expected counts median {np.median(expected):.1f}",
               ylabel="Bq/mL", xlabel="Minutes from PBR28 scan start")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        print(f"offset={offset:g} min | frames={len(mid)} | "
              f"median expected counts={np.median(expected):.2f} | "
              f"median sqrt(N)/N={np.median(1 / np.sqrt(np.maximum(expected, 1))):.3g} | "
              f"sampled total counts={counts.sum()}")
    fig.savefig(output_dir / "02_physical_mixtures.png", dpi=170)
    fig_noise.savefig(output_dir / "03_noise_comparison.png", dpi=170)
    plt.close(fig)
    plt.close(fig_noise)

    # Plot physical decay factors independently as a transparent physics check.
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    t = np.linspace(0, duration_min, 500)
    for tracer in ("FDG", "PBR28"):
        ax.plot(t, np.exp(-np.log(2) * t / HALF_LIVES_MIN[tracer]),
                label=f"{tracer} decay from its own reference time")
    ax.set(xlabel="Minutes since each tracer's reference scan start",
           ylabel="Physical decay factor", ylim=(0, 1.05))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(output_dir / "04_physical_decay_factors.png", dpi=170)
    plt.close(fig)
    print(f"Saved four QC figures under {output_dir}")
    print("CAUTION: These figures do not validate tissue kinetics, reconstruction noise, "
          "injection/scan synchronization, or metabolite correction.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fdg-csv", type=Path, required=True)
    parser.add_argument("--pbr-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/bqml_qc"))
    parser.add_argument("--duration-min", type=float, default=60.0)
    parser.add_argument("--fdg-offsets-min", type=float, nargs="+", default=[0, 5, 10])
    parser.add_argument("--signal-source", choices=("blood", "tissue-approx"),
                        default="tissue-approx")
    parser.add_argument("--effective-sensitivity-cps-per-bq", type=float, default=0.001)
    parser.add_argument("--fdg-kinetics", type=float, nargs=4,
                        metavar=("K1", "k2", "k3", "k4"),
                        help="FDG tissue K1,k2,k3,k4 (per minute)")
    parser.add_argument("--fdg-vb", type=float, default=0.0)
    parser.add_argument("--fdg-delay-sec", type=float, default=0.0)
    args = parser.parse_args()
    plot_validation(args.fdg_csv, args.pbr_csv, args.output_dir,
                    duration_min=args.duration_min,
                    offsets_min=tuple(args.fdg_offsets_min),
                    signal_source=args.signal_source,
                    effective_sensitivity_cps_per_bq=args.effective_sensitivity_cps_per_bq,
                    fdg_kinetics=(dict(zip(("K1", "k2", "k3", "k4"), args.fdg_kinetics))
                                  if args.fdg_kinetics is not None else None),
                    fdg_vb=args.fdg_vb, fdg_delay_sec=args.fdg_delay_sec)


if __name__ == "__main__":
    main()
