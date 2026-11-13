"""Parameterized dual-tracer FDG/PBR28 forward simulator for the EXISTING TCN.

Two workflows:
  1. fit-aif: fit a smooth analytic arterial input model to measured AIF
     frame averages (fit is only for calibration; no measured TAC interpolation).
  2. generate: use JSON model parameters + real TimeStart/TimeEnd CSV to generate
     tracer tissue curves by a continuous-time 2TCM (integrated on a 1-second grid),
     frame-average using exactly the measured frame boundaries, and simulate noise.

All activity curves in Bq/mL. This is a synthetic training dataset, not a
participant-independent validation set. PBR28 whole blood input is a surrogate,
not metabolite-corrected parent plasma.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from sim.bqml_calibration import MCI_TO_MBQ, HALF_LIVES_MIN, REFERENCE_DOSES_MCI
from sim.generate_roi_tcn_dataset import sample_noise

CHANNELS=("FDG","PBR28")


def read_frames(path, duration_min=None):
    with Path(path).open(encoding="utf-8-sig",newline="") as fp:
        rows=list(csv.DictReader(fp))
    if not rows or not {"TimeStart","TimeEnd"}.issubset(rows[0]):
        raise ValueError("Frame CSV must contain TimeStart and TimeEnd (seconds)")
    values=np.asarray([[float(row["TimeStart"]),float(row["TimeEnd"])] for row in rows])
    if (not np.isfinite(values).all() or np.any(values[:,1]<=values[:,0])
            or abs(values[0,0])>1e-5 or np.any(np.diff(values[:,0])<=0)
            or not np.allclose(values[1:,0],values[:-1,1],atol=1e-4)):
        raise ValueError("Frame intervals must be finite, contiguous, increasing, and begin at zero")
    if duration_min is not None:
        if duration_min<=0 or duration_min*60>values[-1,1]+1e-7:
            raise ValueError("Duration outside frame schedule")
        values=values[values[:,0]<duration_min*60-1e-7]
        values[-1,1]=min(values[-1,1],duration_min*60)
    return values/60


def analytic_aif(t_min, cfg):
    """Smooth bolus plus two nonnegative exponential tails, time in minutes.

    c(t) = B (u/peak)^shape exp(shape*(1-u/peak)) +
           A_fast exp(-lambda_fast*u) +
           A_slow exp(-lambda_slow*u), u=t-delay>=0.
    At u=0 tail components are nonzero; enforce an injection switch at t=delay.
    All amplitudes are Bq/mL, rates 1/min.
    """
    t=np.asarray(t_min,dtype=float)
    req=("delay_min","bolus_amplitude_bqml","bolus_peak_min","bolus_shape",
         "fast_amplitude_bqml","fast_lambda_per_min",
         "slow_amplitude_bqml","slow_lambda_per_min")
    if set(req)-set(cfg):
        raise ValueError(f"AIF missing parameters: {sorted(set(req)-set(cfg))}")
    v={k:float(cfg[k]) for k in req}
    if (not np.isfinite(list(v.values())).all() or v["delay_min"]<0
            or v["bolus_peak_min"]<=0 or v["bolus_shape"]<=0
            or any(v[k]<0 for k in req if "amplitude" in k)
            or v["fast_lambda_per_min"]<=0 or v["slow_lambda_per_min"]<=0):
        raise ValueError("Invalid analytic AIF parameters")
    u=np.maximum(t-v["delay_min"],0)
    bolus=v["bolus_amplitude_bqml"]*np.power(
        u/v["bolus_peak_min"],v["bolus_shape"])*np.exp(
        v["bolus_shape"]*(1-u/v["bolus_peak_min"]))
    tail=v["fast_amplitude_bqml"]*np.exp(-v["fast_lambda_per_min"]*u)
    tail+=v["slow_amplitude_bqml"]*np.exp(-v["slow_lambda_per_min"]*u)
    return np.where(t>=v["delay_min"],bolus+tail,0.)


def tcm_curve(cp, dt_min, kin, vb):
    if (set(kin)!={"K1","k2","k3","k4"} or
            any(not np.isfinite(float(v)) or float(v)<0 for v in kin.values())
            or not 0<=vb<1):
        raise ValueError("Invalid 2TCM parameters or vascular fraction")
    cp=np.asarray(cp,float)
    c1=np.zeros_like(cp); c2=np.zeros_like(cp)
    k1,k2,k3,k4=(float(kin[k]) for k in ("K1","k2","k3","k4"))
    # RK4 for the linear system avoids substantial Euler error.
    def deriv(a,b,drive):
        return k1*drive-(k2+k3)*a+k4*b,k3*a-k4*b
    for i in range(1,len(cp)):
        u0,u1=cp[i-1],cp[i]
        d1a,d1b=deriv(c1[i-1],c2[i-1],u0)
        d2a,d2b=deriv(c1[i-1]+.5*dt_min*d1a,c2[i-1]+.5*dt_min*d1b,.5*(u0+u1))
        d3a,d3b=deriv(c1[i-1]+.5*dt_min*d2a,c2[i-1]+.5*dt_min*d2b,.5*(u0+u1))
        d4a,d4b=deriv(c1[i-1]+dt_min*d3a,c2[i-1]+dt_min*d3b,u1)
        c1[i]=c1[i-1]+dt_min*(d1a+2*d2a+2*d3a+d4a)/6
        c2[i]=c2[i-1]+dt_min*(d1b+2*d2b+2*d3b+d4b)/6
    return (1-vb)*(c1+c2)+vb*cp


def dense_frame_means(times,values,frames):
    result=np.empty(len(frames))
    for i,(a,b) in enumerate(frames):
        intermediate=times[(times>a)&(times<b)]
        points=np.r_[a,intermediate,b]
        result[i]=np.trapezoid(np.interp(points,times,values),points)/(b-a)
    return result


def _fit_model_from_frame_averages(path):
    frames=read_frames(path)
    with Path(path).open(encoding="utf-8-sig",newline="") as fp:
        rows=list(csv.DictReader(fp))
    if "aorta" not in rows[0]:
        raise ValueError("AIF fitting CSV needs an aorta column in Bq/mL")
    data=np.array([float(x["aorta"]) for x in rows])
    if (not np.isfinite(data).all() or np.any(data<0)
            or not np.any(data>0)):
        raise ValueError("Invalid observed aortic TAC")
    # Fit in frame-average domain, not midpoint interpolation.
    t=np.linspace(0,frames[-1,1],int(np.ceil(frames[-1,1]*60))+1)
    positive=np.maximum(data,1)
    peak=float(frames[np.argmax(data)].mean())
    initial=np.array([max(.01,min(peak/2,.5)),np.max(data),max(peak,.15),2.,
                      max(data[0],np.max(data)*.2),.6,max(data[-1],1.),.02])
    lower=np.array([0,0,.02,.5,0,.01,0,.0001])
    upper=np.array([min(3.,frames[-1,1]*.2),np.max(data)*10,
                    4.,8.,np.max(data)*10,10.,
                    np.max(data)*10,.3])
    initial=np.minimum(np.maximum(initial,lower+1e-8),upper-1e-8)
    keys=("delay_min","bolus_amplitude_bqml","bolus_peak_min","bolus_shape",
          "fast_amplitude_bqml","fast_lambda_per_min",
          "slow_amplitude_bqml","slow_lambda_per_min")
    def estimate(x):
        return dense_frame_means(t,analytic_aif(t,dict(zip(keys,x))),frames)
    def residual(x):
        return (estimate(x)-data)/np.sqrt(positive)
    fitted=least_squares(residual,initial,bounds=(lower,upper),max_nfev=300)
    config=dict(zip(keys,map(float,fitted.x)))
    prediction=estimate(fitted.x)
    rmse=float(np.sqrt(np.mean((prediction-data)**2)))
    return config,rmse,frames


def fit_aifs(fdg_csv,pbr_csv,out_file):
    fdg,fdg_rmse,_=_fit_model_from_frame_averages(fdg_csv)
    pbr,pbr_rmse,_=_fit_model_from_frame_averages(pbr_csv)
    cfg={
        "aif":{"FDG":fdg,"PBR28":pbr},
        "kinetics":{
            "FDG":{"K1":.146969833218802,"k2":.119575138861569,
                   "k3":.042108367298586,"k4":0.},
            "PBR28":{"K1":.18,"k2":.25,"k3":.04,"k4":.01}},
        "vascular_fraction":{"FDG":.0808523133568341,"PBR28":0.05},
        "reference_doses_mci":{"FDG":4.91,"PBR28":4.98},
        "dose_ranges_mci":{"FDG":[4.91,4.91],"PBR28":[4.98,4.98]},
        "kinetic_variation_fraction":0.1,
        "aif_variation_fraction":0.05,
        "fdg_extra_delay_sec":5.78091827383968,
        "provenance":{
            "aif":"Frame-average analytical model fitted separately to measured blood AIF files",
            "aif_fit_rmse_bqml":{"FDG":fdg_rmse,"PBR28":pbr_rmse},
            "FDG_kinetics":"Previously supplied single-region fitted FDG parameters; not population validated",
            "PBR28_kinetics":"Illustrative parameter defaults, NOT fitted to current subject",
            "PBR28_aif":"Whole-blood AIF surrogate, not metabolite-corrected plasma",
            "reference_doses":"Measured injected activities recalled from prior scan (mCi)",
            "dose_scale_assumption":"Concentrations linear in administered activity",
            "delay_sign":"Positive FDG extra delay shifts tissue input later; original fitter sign not verified"
        }}
    out_file=Path(out_file);out_file.parent.mkdir(parents=True,exist_ok=True)
    out_file.write_text(json.dumps(cfg,indent=2)+"\n")
    print(f"Saved fitted-analytic AIF + editable 2TCM configuration: {out_file}")
    print(f"Fit RMSE: FDG={fdg_rmse:.2f}, PBR28={pbr_rmse:.2f} Bq/mL")
    return cfg


def forward(cfg,frames,offset,fdg_dose,pbr_dose,variation,rng,
            aif_variation=0.,dt_sec=1.,return_aif_phys=False):
    if dt_sec<=0:raise ValueError("dt_sec must be positive")
    duration=frames[-1,1]
    # Exactly uniform integration grid; ensure the final frame edge is a node.
    n_steps=max(1,int(np.ceil(duration*60/dt_sec)))
    t=np.linspace(0,duration,n_steps+1)
    step_min=duration/n_steps
    targets=[];idifs=[]; dc_targets=[]; aif_phys=[]
    for tracer,dose in (("FDG",fdg_dose),("PBR28",pbr_dose)):
        relative=t-(offset if tracer=="FDG" else 0.)
        aif=dict(cfg["aif"][tracer])
        if aif_variation:
            for name in ("bolus_amplitude_bqml","fast_amplitude_bqml","slow_amplitude_bqml"):
                aif[name]*=rng.uniform(1-aif_variation,1+aif_variation)
        reference=float(cfg["reference_doses_mci"][tracer])
        if reference<=0:raise ValueError("reference doses must be positive")
        plasma_like=analytic_aif(relative,aif)*dose/reference
        extra_delay=float(cfg.get("fdg_extra_delay_sec",0.)) if tracer=="FDG" else 0.
        drive=(analytic_aif(relative-extra_delay/60,aif)*dose/reference
               if extra_delay else plasma_like)
        k={x:float(v)*rng.uniform(1-variation,1+variation)
           if variation else float(v)
           for x,v in cfg["kinetics"][tracer].items()}
        vb=float(cfg["vascular_fraction"][tracer])
        tissue_dc=tcm_curve(drive,step_min,k,vb)
        decay=np.exp(-np.log(2)*np.maximum(relative,0)/HALF_LIVES_MIN[tracer])
        targets.append(dense_frame_means(t,tissue_dc*decay,frames))
        dc_targets.append(dense_frame_means(t,tissue_dc,frames))
        idifs.append(dense_frame_means(t,plasma_like,frames))
        aif_phys.append(dense_frame_means(t,plasma_like*decay,frames))
    result=(np.asarray(targets),np.asarray(dc_targets),np.asarray(idifs))
    return result+(np.asarray(aif_phys),) if return_aif_phys else result


def generate(params_path,frames_path,output_dir,*,n_samples=1000,seed=48,
             duration_min=60.,offsets=(0,5,10),cv60=.05,dt_sec=1.,
             train_fraction=.8,val_fraction=.1,target_mode="tissue"):
    if target_mode not in ("tissue","aif"):
        raise ValueError("target_mode must be tissue or aif")
    cfg=json.loads(Path(params_path).read_text())
    frames=read_frames(frames_path,duration_min)
    if (n_samples<10 or not 0<train_fraction<1
            or not 0<val_fraction<1 or train_fraction+val_fraction>=1):
        raise ValueError("Need at least 10 samples and valid split fractions")
    if cv60<=0 or not np.isfinite(cv60):
        raise ValueError("Invalid ROI-average CV")
    if not offsets or any(x<0 for x in offsets):
        raise ValueError("Invalid FDG injection offsets")
    variation=float(cfg.get("kinetic_variation_fraction",0.))
    aif_variation=float(cfg.get("aif_variation_fraction",0.))
    if not (0<=variation<1 and 0<=aif_variation<1):
        raise ValueError("Variation fractions must be within [0,1)")
    doses=cfg["dose_ranges_mci"]
    for tracer in CHANNELS:
        if any(float(v)<=0 for v in doses[tracer]) or doses[tracer][0]>doses[tracer][1]:
            raise ValueError("Invalid administered activity range")
    rng=np.random.default_rng(seed)
    samples=[]
    for i in range(n_samples):
        offset=float(rng.choice(offsets))
        fdg_dose=float(rng.uniform(*doses["FDG"]))
        pbr_dose=float(rng.uniform(*doses["PBR28"]))
        tissue_target,dc_target,idifs,aif_target=forward(
            cfg,frames,offset,fdg_dose,pbr_dose,
            variation,rng,aif_variation,dt_sec,return_aif_phys=True)
        target=tissue_target if target_mode=="tissue" else aif_target
        clean=target.sum(axis=0)
        observed,expected=sample_noise(clean,frames[:,1]-frames[:,0],cv60,rng)
        samples.append({
            "X_measured_tavg_n1_d1":observed[None,:].astype("float32"),
            "X_clean_tavg_n0_d1":clean[None,:].astype("float32"),
            "Y_phys_tavg_n0_d1":target.astype("float32"),
            "Y_bio_tavg_n0_d0":(dc_target if target_mode=="tissue" else idifs).astype("float32"),
            "Y_tissue_phys_tavg_n0_d1":tissue_target.astype("float32"),
            "Y_aif_phys_tavg_n0_d1":aif_target.astype("float32"),
            "expected_counts":expected.astype("float32"),
            "fdg_idif_dc_bqml":idifs[0].astype("float32"),
            "pbr28_idif_dc_bqml":idifs[1].astype("float32"),
            "metadata":{
                "sample_id":i,"delta_min":offset,
                "fdg_dose_mci":fdg_dose,"pbr28_dose_mci":pbr_dose,
                "fdg_dose_mbq":fdg_dose*MCI_TO_MBQ,
                "pbr28_dose_mbq":pbr_dose*MCI_TO_MBQ,
                "tracer2_scale":fdg_dose/float(cfg["reference_doses_mci"]["FDG"]),
                "tracer1_scale":pbr_dose/float(cfg["reference_doses_mci"]["PBR28"])
            }
        })
    order=rng.permutation(n_samples)
    a=int(train_fraction*n_samples);b=a+int(val_fraction*n_samples)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    for split,indices in (("train",order[:a]),("val",order[a:b]),("test",order[b:])):
        keys=("X_measured_tavg_n1_d1","X_clean_tavg_n0_d1",
              "Y_phys_tavg_n0_d1","Y_bio_tavg_n0_d0",
              "expected_counts","fdg_idif_dc_bqml","pbr28_idif_dc_bqml",
              "Y_tissue_phys_tavg_n0_d1","Y_aif_phys_tavg_n0_d1")
        np.savez_compressed(out/f"{split}.npz",
                            **{k:np.stack([samples[int(i)][k] for i in indices])
                               for k in keys})
        (out/f"{split}_metadata.json").write_text(json.dumps(
            [samples[int(i)]["metadata"] for i in indices],indent=2)+"\n")
    meta={
        "units":{"TAC":"Bq/mL","time":"minutes"},
        "frame_source":str(frames_path),
        "frame_edges_min":np.r_[frames[0,0],frames[:,1]].tolist(),
        "frame_midpoints_min":frames.mean(axis=1).tolist(),
        "frame_durations_min":(frames[:,1]-frames[:,0]).tolist(),
        "params_file":str(params_path),
        "reference_doses_mci":cfg["reference_doses_mci"],
        "fdg_offsets_min":list(offsets),
        "source_signal":("analytic AIF -> 2TCM tissue forward simulation" if target_mode=="tissue" else "analytic AIF physical blood activity separation"),
        "target_mode":target_mode,
        "aif_fit_inputs":"analytic AIF calibrated using measured frame averages, but samples generated from parametric curves",
        "noise":{"model":"effective Poisson for ROI mean","cv_60s_at_10k_bqml":cv60,
                 "warning":"Illustrative unless independently ROI-calibrated"},
        "seed":seed,"sample_count":n_samples,"channel_order":list(CHANNELS),
        "validation_warning":"Same AIF parameter sources across splits: synthetic variation, not subject-independent data"
    }
    (out/"dataset_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(f"Generated {n_samples} model-based TCN samples, {len(frames)} measured acquisition frames -> {out}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    fit=sub.add_parser("fit-aif",help="Fit analytical AIF parameters using measured frame-averaged aorta data")
    fit.add_argument("--fdg-aif-csv",type=Path,required=True)
    fit.add_argument("--pbr-aif-csv",type=Path,required=True)
    fit.add_argument("--out-params",type=Path,required=True)
    gen=sub.add_parser("generate",help="Generate a TCN dataset entirely from model parameters")
    gen.add_argument("--params",type=Path,required=True)
    gen.add_argument("--frame-csv",type=Path,required=True)
    gen.add_argument("--output-dir",type=Path,required=True)
    gen.add_argument("--duration-min",type=float,default=60.)
    gen.add_argument("--fdg-offsets-min",type=float,nargs="+",default=[0,5,10])
    gen.add_argument("--noise-cv60",type=float,default=.05)
    gen.add_argument("--n-samples",type=int,default=1000)
    gen.add_argument("--seed",type=int,default=48)
    gen.add_argument("--integration-step-sec",type=float,default=1.)
    gen.add_argument("--target-mode",choices=["tissue","aif"],default="tissue",
                     help="Train target: simulated tissue TACs or independently varied blood AIFs")
    a=parser.parse_args()
    if a.action=="fit-aif":
        fit_aifs(a.fdg_aif_csv,a.pbr_aif_csv,a.out_params)
    else:
        generate(a.params,a.frame_csv,a.output_dir,n_samples=a.n_samples,
                 seed=a.seed,duration_min=a.duration_min,
                 offsets=tuple(a.fdg_offsets_min),cv60=a.noise_cv60,
                 dt_sec=a.integration_step_sec,target_mode=a.target_mode)


if __name__=="__main__":
    main()
