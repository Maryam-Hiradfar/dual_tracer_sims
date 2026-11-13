"""Generate the EXISTING TCN's train/val/test NPZ schema from measured brain ROI TACs.

No phantom, NNLS, or alternative prediction model. FDG/PBR columns should
be paired by identical name or an explicit mapping CSV. All signals in Bq/mL.
Noise calibrated from a representative ROI via a measured voxel CV + a
user-specified effective independent voxel count (NOT raw voxel count).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sim.roi_pilot import load_rois, resolve_rois, frame_activity, preferred_frame_edges, HALF_LIVES


def calibrate_reference_cv(path: Path, effective_roi_voxels: float,
                           reference_bqml: float=10000., reference_seconds: float=60.):
    """Map measured voxel spatial-residual CV to an *assumed* ROI-average CV.

    The effective independent voxel count needs independent justification:
    voxel correlations mean it is much smaller than physical voxel count.
    """
    df=pd.read_csv(path)
    required={"cv","mean_bqml","TimeStart","TimeEnd"}
    if not required.issubset(df.columns):
        raise ValueError(f"noise calibration CSV requires columns {sorted(required)}")
    if effective_roi_voxels<=0 or reference_bqml<=0 or reference_seconds<=0:
        raise ValueError("effective count, reference activity and duration must be positive")
    dt=(df.TimeEnd-df.TimeStart).to_numpy(float)
    concentration=df.mean_bqml.to_numpy(float)
    cv=df.cv.to_numpy(float)
    keep=np.isfinite(cv)&(cv>0)&np.isfinite(concentration)&(concentration>0)&(dt>0)
    if not np.any(keep):
        raise ValueError("no valid noise CV rows")
    normalized = cv[keep]*np.sqrt(concentration[keep]/reference_bqml *
                                  dt[keep]/reference_seconds)
    cv60=float(np.median(normalized)/np.sqrt(effective_roi_voxels))
    return cv60, dict(n_calibration_frames=int(keep.sum()),
                      effective_independent_roi_voxels=effective_roi_voxels,
                      median_voxel_cv60_at_reference=float(np.median(normalized)),
                      reference_concentration_bqml=reference_bqml,
                      reference_duration_s=reference_seconds,
                      calibration_csv=str(path))


def sample_noise(clean, widths, cv60, rng, reference_bqml=10000.):
    """Effective count-domain stochastic ROI noise, seeded and physically scaled."""
    coeff=1/(cv60**2*reference_bqml*60.)
    conversion=np.asarray(widths)*60*coeff
    lam=np.maximum(np.asarray(clean),0)*conversion
    return rng.poisson(lam)/conversion,lam




def measured_frame_edges(df, duration_min):
    """Use original TimeStart/TimeEnd boundaries (seconds) without inventing frames."""
    if not {"TimeStart", "TimeEnd"}.issubset(df.columns):
        raise ValueError("--frame-schedule measured requires TimeStart and TimeEnd columns "
                         "in the PBR28 source CSV")
    starts = df["TimeStart"].to_numpy(float) / 60
    ends = df["TimeEnd"].to_numpy(float) / 60
    if abs(starts[0]) > 1e-6 or np.any(ends <= starts):
        raise ValueError("invalid measured PBR28 framing")
    if duration_min > ends[-1] + 1e-8:
        raise ValueError("requested duration exceeds measured PBR28 frames")
    # Do not silently bridge missing acquisition intervals in an edge-based model.
    if not np.allclose(starts[1:], ends[:-1], atol=1e-5):
        raise ValueError("PBR28 timing has gaps or overlaps; cannot represent as contiguous frame edges")
    cut = np.searchsorted(ends, duration_min - 1e-9)
    edges = np.r_[starts[0], ends[:cut+1]]
    # A final partial frame is permitted only if the requested duration
    # falls inside an existing measured frame.
    if edges[-1] > duration_min + 1e-8:
        edges[-1] = duration_min
    if np.any(np.diff(edges) <= 0):
        raise ValueError("invalid measured frame edges")
    return edges

def build(fdg_path,pbr_path,output_dir,*,mapping=None,rois=(),duration_min=60.,
          offsets=(0.,5.,10.),n_samples=1000,seed=48,
          cv60=None,noise_csv=None,effective_roi_voxels=None,
          dose_scale=(1.,1.),frame_schedule="measured",training_fraction=.8,val_fraction=.1):
    fdg,ft,fc,fdg_cols=load_rois(fdg_path)
    pbr,pt,pc,pbr_cols=load_rois(pbr_path)
    pairs=resolve_rois(fdg_cols,pbr_cols,rois,mapping)
    offsets=tuple(float(v) for v in offsets)
    if not pairs or not offsets or min(offsets)<0:
        raise ValueError("need ROI pairs and nonnegative offsets")
    if duration_min>pc or any(duration_min-offset>fc for offset in offsets):
        raise ValueError("requested duration exceeds recorded ROI TAC coverage")
    if n_samples<10 or not (0<training_fraction<1 and 0<val_fraction<1 and training_fraction+val_fraction<1):
        raise ValueError("need >=10 samples and valid train/val/test fractions")
    if dose_scale[0]<=0 or dose_scale[1]<dose_scale[0]:
        raise ValueError("invalid dose scaling range")
    if (noise_csv is None)==(cv60 is None):
        raise ValueError("supply exactly one of --noise-samples-csv or --noise-cv60")
    if noise_csv is not None:
        if effective_roi_voxels is None:
            raise ValueError("--effective-roi-voxels is required for voxel-CV calibration")
        cv60,noise_meta=calibrate_reference_cv(noise_csv,effective_roi_voxels)
    else:
        noise_meta={"source":"manually supplied ROI-average CV at 10k Bq/mL and 60s; uncalibrated"}
    if cv60<=0 or not np.isfinite(cv60):
        raise ValueError("invalid reference ROI noise CV")
    if frame_schedule not in ("measured", "preferred"):
        raise ValueError("frame_schedule must be measured or preferred")
    edges=(measured_frame_edges(pbr,duration_min) if frame_schedule=="measured"
           else preferred_frame_edges(duration_min))
    widths=np.diff(edges)
    rng=np.random.default_rng(seed)
    templates={}
    for name,fcname,pcname in pairs:
        for offset in offsets:
            f=frame_activity(ft,fdg[fcname].to_numpy(float),edges,offset,HALF_LIVES["FDG"])
            p=frame_activity(pt,pbr[pcname].to_numpy(float),edges,0.,HALF_LIVES["PBR28"])
            templates[(name,offset)]=np.stack((f,p))
    samples=[]
    for i in range(n_samples):
        region,_,_=pairs[int(rng.integers(len(pairs)))]
        delay=offsets[int(rng.integers(len(offsets)))]
        fdg_scale,pbr_scale=(1.,1.) if dose_scale==(1.,1.) else rng.uniform(*dose_scale,size=2)
        target=templates[(region,delay)]*np.array([[fdg_scale],[pbr_scale]])
        clean=target.sum(axis=0)
        measured,expected=sample_noise(clean,widths,cv60,rng)
        samples.append(dict(
            X_measured_tavg_n1_d1=measured[None,:].astype(np.float32),
            X_clean_tavg_n0_d1=clean[None,:].astype(np.float32),
            Y_phys_tavg_n0_d1=target.astype(np.float32),
            metadata=dict(sample_id=i,roi=region,delta_min=delay,
                          tracer1_scale=pbr_scale,tracer2_scale=fdg_scale,
                          fdg_scale=fdg_scale,pbr28_scale=pbr_scale),
            expected_counts=expected.astype(np.float32)))
    order=rng.permutation(n_samples)
    train_end=int(n_samples*training_fraction)
    val_end=train_end+int(n_samples*val_fraction)
    splits=dict(train=order[:train_end],val=order[train_end:val_end],test=order[val_end:])
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    for split,ids in splits.items():
        if len(ids)==0: raise ValueError(f"empty {split} split")
        np.savez_compressed(output_dir/f"{split}.npz",
            **{k:np.stack([samples[int(i)][k] for i in ids])
               for k in ("X_measured_tavg_n1_d1","X_clean_tavg_n0_d1",
                         "Y_phys_tavg_n0_d1","expected_counts")})
        (output_dir/f"{split}_metadata.json").write_text(json.dumps(
            [samples[int(i)]["metadata"] for i in ids],indent=2)+"\n")
    (output_dir/"dataset_metadata.json").write_text(json.dumps(dict(
        units={"TAC":"Bq/mL","time":"minutes"},
        frame_edges_min=edges.tolist(),
        frame_schedule_source=("PBR28 TimeStart/TimeEnd" if frame_schedule=="measured" else "preferred"),
        original_fdg_frame_starts_sec=(fdg["TimeStart"].to_numpy(float).tolist() if "TimeStart" in fdg else None),
        original_pbr_frame_starts_sec=(pbr["TimeStart"].to_numpy(float).tolist() if "TimeStart" in pbr else None),
        fdg_scaling_range=list(dose_scale),pbr28_scaling_range=list(dose_scale),
        frame_midpoints_min=((edges[1:]+edges[:-1])/2).tolist(),
        frame_durations_min=widths.tolist(),
        selected_rois=[p[0] for p in pairs],
        source_fdg=str(fdg_path),source_pbr28=str(pbr_path),
        fdg_offsets_min=list(offsets),seed=seed,sample_count=n_samples,
        calibrated_reference_roi_cv60=float(cv60),calibration_details=noise_meta,
        channel_order=["FDG","PBR28"],
        physical_decay_applied=True,
        caveats=["This is a data-augmented synthetic benchmark from measured single-tracer TACs.",
                 "Random train/val/test variants of the same ROIs share source templates; NOT subject-independent validation.",
                 "FDG is rebinned onto PBR28 acquisition frames; interpolation of original frame means is approximate.",
                 "No random amplitude scaling by default: measured Bq/mL levels are retained.",
                 "ROI noise model assumes effective Poisson statistics and equivalent independent voxel count.",
                 "Source TACs assumed individually decay-corrected to injection/scan start."]
    ),indent=2)+"\n")
    print(f"Created TCN-ready data in {output_dir}: {len(pairs)} ROI(s), "
          f"{len(edges)-1} frames, noise reference CV60={cv60:.3%}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fdg-rois",type=Path,required=True)
    p.add_argument("--pbr-rois",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--roi-mapping",type=Path)
    p.add_argument("--rois",nargs="+",default=[])
    p.add_argument("--duration-min",type=float,default=60)
    p.add_argument("--frame-schedule",choices=["measured","preferred"],default="measured",
                   help="Use original PBR28 TimeStart/TimeEnd (default) or preferred frame schedule")
    p.add_argument("--scale-range",type=float,nargs=2,metavar=("LOW","HIGH"),default=[1.,1.],
                   help="Optional independent FDG/PBR28 amplitude augmentation, default 1 1")
    p.add_argument("--offsets-min",type=float,nargs="+",default=[0,5,10])
    p.add_argument("--noise-samples-csv",type=Path)
    p.add_argument("--effective-roi-voxels",type=float)
    p.add_argument("--noise-cv60",type=float,help="ROI-average CV at 10k Bq/mL/60 seconds")
    p.add_argument("--n-samples",type=int,default=1000)
    p.add_argument("--seed",type=int,default=48)
    a=p.parse_args()
    build(a.fdg_rois,a.pbr_rois,a.output_dir,mapping=a.roi_mapping,
          rois=a.rois,duration_min=a.duration_min,offsets=a.offsets_min,
          cv60=a.noise_cv60,noise_csv=a.noise_samples_csv,
          effective_roi_voxels=a.effective_roi_voxels,
          n_samples=a.n_samples,seed=a.seed,
          frame_schedule=a.frame_schedule,dose_scale=tuple(a.scale_range))


if __name__=="__main__":
    main()
