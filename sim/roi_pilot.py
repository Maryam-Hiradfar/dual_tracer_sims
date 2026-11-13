"""Measured ROI TAC dual-tracer pilot: frame rebinning, delay/noise sweeps, slide figure.

No DICOM, atlas, or phantom required. Human-derived inputs/outputs remain local.
Inputs: FDG and PBR28 CSV/XLSX, TimeMid seconds, ROI columns in Bq/mL,
decay-corrected to individual scan starts. Prefer TimeStart/TimeEnd too.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

HALF_LIVES = {"FDG": 109.8, "PBR28": 20.4}


def preferred_frame_edges(duration_min=60.0):
    """Earlier project framing schedule, explicitly truncated at duration."""
    durations_s = ([2] * 20 + [5] * 12 + [10] * 12 + [20] * 12
                   + [30] * 8 + [60] * 6 + [120] * 5
                   + [300] * 8 + [600] * 4)
    edges = np.r_[0., np.cumsum(durations_s) / 60]
    if duration_min <= 0 or duration_min > edges[-1]:
        raise ValueError("requested duration outside preferred frame schedule")
    return np.r_[edges[edges < duration_min - 1e-9], duration_min]


def load_rois(path):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")
    if "TimeMid" not in df:
        raise ValueError(f"{path} must have TimeMid in seconds")
    t = pd.to_numeric(df["TimeMid"], errors="raise").to_numpy(dtype=float) / 60
    if len(t) < 3 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("TimeMid must be finite, increasing, with >=3 samples")
    if {"TimeStart", "TimeEnd"}.issubset(df.columns):
        start = pd.to_numeric(df["TimeStart"], errors="raise").to_numpy(float) / 60
        end = pd.to_numeric(df["TimeEnd"], errors="raise").to_numpy(float) / 60
        if (abs(start[0]) > 1e-6 or np.any(end <= start)
                or np.any(np.diff(start) <= 0) or np.any(start[1:] < end[:-1] - 1e-5)):
            raise ValueError("invalid input frame boundaries")
        coverage = float(end[-1])
    else:
        # A midpoint-only input cannot establish the end of the final frame.
        coverage = float(t[-1])
    ignore = {"TimeStart", "TimeEnd", "TimeMid", "aorta", "Cp"}
    names = [c for c in df.columns if c not in ignore]
    if not names:
        raise ValueError("no ROI TAC columns")
    for c in names:
        v = pd.to_numeric(df[c], errors="raise").to_numpy(float)
        if np.any(~np.isfinite(v)) or np.any(v < 0):
            raise ValueError(f"negative/nonfinite TAC data in ROI {c}")
    return df, t, coverage, names


def frame_activity(t_mid, values, edges, injection_min, half_life_min):
    """Approximate frame mean from interpolated, physically decaying TAC.

    Extrapolation beyond measured scan coverage must be checked by caller.
    """
    values = np.asarray(values, float)
    if injection_min < 0 or len(t_mid) != len(values):
        raise ValueError("invalid injection or TAC length")
    out = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= injection_min:
            out.append(0.)
            continue
        # Include injection time, source midpoint knots, frame edges and 1s grid.
        lo = max(start, injection_min)
        pts = np.unique(np.r_[
            lo, end, t_mid[(t_mid + injection_min > lo) &
                           (t_mid + injection_min < end)] + injection_min,
            np.arange(lo, end, 1 / 60)
        ])
        relative = pts - injection_min
        profile = np.interp(relative, t_mid, values)
        # Ramp from zero to first measured frame midpoint, rather than
        # assigning the first frame average at the instant of injection.
        profile = np.where(relative < t_mid[0],
                           values[0] * np.maximum(relative, 0) / t_mid[0],
                           profile)
        activity = profile * np.exp(-np.log(2) * relative / half_life_min)
        out.append(float(np.trapezoid(activity, pts) / (end - start)))
    return np.asarray(out)


def resolve_rois(fdg_cols, pbr_cols, requested=(), mapping=None):
    """Return (display_name, fdg_col, pbr_col); require explicit unambiguous matches."""
    if mapping is not None:
        pairs = pd.read_csv(mapping)
        if not {"region_name", "fdg_column", "pbr_column"}.issubset(pairs.columns):
            raise ValueError("mapping CSV needs region_name,fdg_column,pbr_column")
        chosen = []
        for row in pairs.itertuples(index=False):
            if pd.isna(row.fdg_column) or pd.isna(row.pbr_column):
                continue
            if row.fdg_column not in fdg_cols or row.pbr_column not in pbr_cols:
                raise ValueError(f"mapping references absent columns for {row.region_name}")
            if not requested or row.region_name in requested:
                chosen.append((str(row.region_name), str(row.fdg_column), str(row.pbr_column)))
        if requested and set(requested) - {x[0] for x in chosen}:
            raise ValueError("some requested ROI names absent from mapping")
        return chosen
    common = [c for c in fdg_cols if c in pbr_cols]
    if requested:
        missing = set(requested) - set(common)
        if missing:
            raise ValueError(f"ROI columns not matched in both tables: {sorted(missing)}")
        common = [c for c in requested]
    if not common:
        raise ValueError("no identically named ROIs: supply --roi-mapping")
    return [(str(c), c, c) for c in common]


def sample_roi_noise(clean, frame_dur_min, rng, cv_60s=0.05,
                     concentration_reference=10000.0):
    """Exploratory effective Poisson noise for an ROI-averaged measurement.

    CV scales approximately as (activity x frame duration)^(-1/2).
    The reference CV is NOT a measured GS voxel noise level.
    """
    if cv_60s <= 0 or concentration_reference <= 0:
        raise ValueError("positive noise CV and reference concentration required")
    # For reference 60-s, 10k Bq/mL activity: N_reference=1/CV^2.
    equivalent = 1 / (cv_60s**2 * concentration_reference * 60)
    conversion = np.asarray(frame_dur_min) * 60 * equivalent
    expected = np.maximum(np.asarray(clean), 0) * conversion
    return rng.poisson(expected) / conversion


def nrmse(a, b):
    scale = np.max(b) - np.min(b)
    if scale <= 0:
        scale = np.mean(np.abs(b))
    return float(np.sqrt(np.mean((np.asarray(a) - b) ** 2)) / max(scale, 1e-12))


def run(fdg_path, pbr_path, out_dir, *, duration_min=60.,
        offsets=(0., 5., 10.), rois=(), mapping=None,
        noise_levels=(0.05, 0.10), reps=50, seed=48, preview_rois=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fdg, ft, fc, fdg_cols = load_rois(fdg_path)
    pbr, pt, pc, pbr_cols = load_rois(pbr_path)
    chosen = resolve_rois(fdg_cols, pbr_cols, rois, mapping)
    edges = preferred_frame_edges(duration_min)
    offsets = [float(x) for x in offsets]
    if not offsets or any(x < 0 for x in offsets):
        raise ValueError("offsets must be >= 0")
    if duration_min > pc + 1e-8 or any(duration_min - x > fc + 1e-8 for x in offsets):
        raise ValueError("output extends past measured ROI TAC coverage; shorten --duration-min")
    if reps < 1 or preview_rois < 1:
        raise ValueError("reps/preview-rois must be positive")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    mids = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    measurements = []
    scores = []
    for region, fcol, pcol in chosen:
        for offset in offsets:
            f = frame_activity(ft, fdg[fcol].to_numpy(float), edges, offset, HALF_LIVES["FDG"])
            p = frame_activity(pt, pbr[pcol].to_numpy(float), edges, 0., HALF_LIVES["PBR28"])
            mix = f + p
            if not np.allclose(mix, f+p):
                raise AssertionError("physical mixture must be component sum")
            for idx in range(len(mids)):
                measurements.append(dict(roi=region, fdg_column=fcol, pbr_column=pcol,
                    offset_min=offset, frame_index=idx, TimeStart=edges[idx]*60,
                    TimeEnd=edges[idx+1]*60, TimeMid=mids[idx]*60,
                    fdg_phys_bqml=f[idx], pbr_phys_bqml=p[idx], mixture_clean_bqml=mix[idx]))
            for cv in noise_levels:
                trials = np.stack([sample_roi_noise(mix, widths, rng, cv) for _ in range(reps)])
                for trial in range(reps):
                    scores.append(dict(roi=region, offset_min=offset, noise_cv60=cv,
                       replicate=trial, mixture_nrmse=nrmse(trials[trial], mix),
                       relative_rmse=float(np.sqrt(np.mean((trials[trial]-mix)**2)) /
                                           max(np.sqrt(np.mean(mix**2)), 1e-12))))
    df = pd.DataFrame(measurements)
    df.to_csv(out_dir/"roi_mixture_tacs.csv",index=False)
    metrics = pd.DataFrame(scores)
    metrics.to_csv(out_dir/"roi_noise_trials.csv",index=False)
    summary = metrics.groupby(["roi","offset_min","noise_cv60"],as_index=False).agg(
        mean_mixture_nrmse=("mixture_nrmse","mean"),
        sd_mixture_nrmse=("mixture_nrmse","std"),
        mean_relative_rmse=("relative_rmse","mean"))
    summary.to_csv(out_dir/"roi_noise_summary.csv",index=False)

    # Figure: three representative ROIs; corresponding FDG / PBR / mixed curves.
    present = chosen[:preview_rois]
    fig, axes = plt.subplots(len(present), 2, figsize=(13, 3.4*len(present)),
                             squeeze=False, constrained_layout=True)
    representative = 5. if 5. in offsets else offsets[0]
    noisy_cv = float(noise_levels[0]) if noise_levels else None
    plot_rng = np.random.default_rng(seed + 911)
    for k, (region, _, _) in enumerate(present):
        d = df[(df.roi == region) & (df.offset_min == representative)]
        for col, label in [("fdg_phys_bqml","FDG reference"),
                           ("pbr_phys_bqml","PBR28 reference"),
                           ("mixture_clean_bqml","Synthetic mixture")]:
            axes[k,0].plot(d.TimeMid/60, d[col], label=label, lw=1.5)
        if noisy_cv is not None:
            observed = sample_roi_noise(d.mixture_clean_bqml.to_numpy(float),
                                        widths, plot_rng, noisy_cv)
            axes[k,0].plot(d.TimeMid/60, observed, ".", alpha=.55, ms=3,
                           label=f"Noisy mixture (CV60={100*noisy_cv:g}%)")
        axes[k,0].set(title=f"{region}: offset={representative:g} min",
                       xlabel="Minutes from PBR28 reference",ylabel="Bq/mL")
        axes[k,0].legend(fontsize=7)
        axes[k,0].grid(alpha=.2)
        for offset in offsets:
            t = df[(df.roi==region)&(df.offset_min==offset)]
            axes[k,1].plot(t.TimeMid/60,t.mixture_clean_bqml,
                           label=f"FDG +{offset:g} min")
        axes[k,1].set(title=f"{region}: injection-delay sensitivity",
                       xlabel="Minutes",ylabel="Mixture Bq/mL")
        axes[k,1].grid(alpha=.2)
        axes[k,1].legend(fontsize=8)
    fig.suptitle("Measured single-tracer ROI TACs -> synthetic physical FDG/PBR28 mixtures",
                 fontsize=15)
    fig.savefig(out_dir/"01_roi_pilot_figure.png",dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,5),constrained_layout=True)
    if not summary.empty:
        for cv, v in summary.groupby("noise_cv60"):
            means = v.groupby("offset_min")["mean_mixture_nrmse"].mean()
            ax.plot(means.index,means.values,"o-",label=f"Reference CV60={cv*100:g}%")
    ax.set(title="Noisy mixture error (not separation error)",
           xlabel="FDG injection offset (min)",ylabel="Mean NRMSE vs clean mixture")
    ax.grid(alpha=.2);ax.legend()
    fig.savefig(out_dir/"02_noise_sensitivity.png",dpi=190)
    plt.close(fig)

    (out_dir/"pilot_metadata.json").write_text(json.dumps({
        "fdg_input":str(fdg_path),"pbr_input":str(pbr_path),
        "units":"Bq/mL", "frame_edges_min":edges.tolist(),"offsets_min":offsets,
        "selected_rois":[x[0] for x in chosen], "noise_cv60":list(noise_levels),
        "noise_reference_concentration_bqml":10000,
        "seed":seed,"reps":reps,
        "limitations":[
            "PBR28 is injected first; FDG occurs at offset",
            "Each source TAC is assumed decay-corrected to its injection/scan start",
            "Frame-mean TAC interpolation approximates continuous tracer kinetics",
            "ROI-level synthetic mixture is not observed simultaneous dual-tracer PET",
            "Noise uses illustrative effective ROI Poisson counts, not GS-calibrated noise",
            "Noise NRMSE measures mixture perturbation, not tracer-separation performance"
        ]},indent=2)+"\n")
    return df,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fdg-rois",type=Path,required=True)
    p.add_argument("--pbr-rois",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,default=Path("results/roi_pilot"))
    p.add_argument("--roi-mapping",type=Path,help="Optional region_name,fdg_column,pbr_column CSV")
    p.add_argument("--rois",nargs="+",default=[])
    p.add_argument("--duration-min",type=float,default=60)
    p.add_argument("--offsets-min",type=float,nargs="+",default=[0,5,10])
    p.add_argument("--noise-cv60",type=float,nargs="+",default=[.05,.10])
    p.add_argument("--noise-reps",type=int,default=50)
    p.add_argument("--preview-rois",type=int,default=3)
    p.add_argument("--seed",type=int,default=48)
    a=p.parse_args()
    _,summary=run(a.fdg_rois,a.pbr_rois,a.output_dir,duration_min=a.duration_min,
        offsets=a.offsets_min,rois=a.rois,mapping=a.roi_mapping,
        noise_levels=a.noise_cv60,reps=a.noise_reps,seed=a.seed,
        preview_rois=a.preview_rois)
    print(f"Saved ROI pilot figure, mixture TACs and noise summaries to {a.output_dir}")
    print(summary.head(12).to_string(index=False))


if __name__=="__main__":
    main()
