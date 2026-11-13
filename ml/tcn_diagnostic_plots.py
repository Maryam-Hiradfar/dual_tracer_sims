"""Comprehensive plots for the EXISTING TCN evaluator.

Uses only predictions/ground truth from saved model and data. Does not fit an
alternative signal separator. Legends live in dedicated margins, never on data.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=240,bbox_inches="tight")
    plt.close(fig)


def _legend_below(fig, axes, ncol=4):
    handles, labels=[],[]
    for ax in np.ravel(axes):
        h,lab=ax.get_legend_handles_labels()
        for x,y in zip(h,lab):
            if y not in labels:
                handles.append(x); labels.append(y)
    if handles:
        fig.legend(handles,labels,loc="lower center",ncol=min(ncol,len(labels)),
                   fontsize=9,frameon=False,bbox_to_anchor=(.5,.005))
    fig.subplots_adjust(bottom=.18,top=.88,wspace=.28,hspace=.35)


def _unit(meta):
    unit=meta.get("units",{}).get("TAC","Bq/mL")
    if unit not in ("Bq/mL","kBq/mL"):
        raise ValueError("Unknown source concentration units")
    return unit


def render_tcn_diagnostics(frame_midpoints, measured, clean, target, prediction,
                           metadata, global_metadata, selected, output_dir,
                           training_history=None):
    """Generate diagnostic figures; no NNLS or extra model fitting."""
    t=np.asarray(frame_midpoints,float)
    x=np.asarray(measured,float)
    c=np.asarray(clean,float)
    y=np.asarray(target,float)
    p=np.asarray(prediction,float)
    if (x.ndim!=3 or y.ndim!=3 or p.shape!=y.shape or x.shape[0]!=len(y)
            or x.shape[1]!=1 or y.shape[1]!=2 or c.shape!=x.shape
            or y.shape[2]!=len(t) or len(metadata)!=len(y)):
        raise ValueError("Incompatible TCN evaluation shapes")
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    unit=_unit(global_metadata)
    mode=global_metadata.get("target_mode")
    kind=("Synthetic arterial input functions (AIFs)" if mode=="aif" else
          "Simulated tissue TACs (2TCM)" if mode=="tissue" else
          "Aortic blood curves (input functions)" if global_metadata.get("selected_rois")==["aorta"] else "ROI TACs")
    sources=("FDG","PBR28")
    images={}

    # The familiar 3-panel mixed-signal view, no annotations over signal.
    fig,axes=plt.subplots(1,len(selected),figsize=(6*len(selected),4.7),
                          squeeze=False)
    axes=axes.ravel()
    for ax,i in zip(axes,selected):
        ax.scatter(t,x[i,0],s=10,color="0.55",alpha=.65,label="Observed mixture")
        ax.plot(t,c[i,0],color="black",lw=1.3,label="Clean mixture")
        for ch,name,color in ((0,"FDG","C0"),(1,"PBR28","C1")):
            ax.plot(t,y[i,ch],color=color,lw=1.8,label=name+" true")
            ax.plot(t,p[i,ch],color=color,lw=1.8,ls="--",label=name+" TCN")
        ax.set(title=f"FDG injection +{metadata[i]['delta_min']:g} min",
               xlabel="Time from PBR28 injection (min)",ylabel=f"Activity ({unit})")
        ax.grid(alpha=.2)
    fig.suptitle(f"TCN dual-tracer separation: {kind}",fontsize=14)
    _legend_below(fig,axes,ncol=6)
    path=out/"test_prediction_examples.png";_save(fig,path)
    images["prediction_examples"]=str(path)

    # Tracer components separately for readable matching of solid/dashed lines.
    fig,axes=plt.subplots(2,len(selected),figsize=(5.7*len(selected),7),
                          squeeze=False)
    for j,i in enumerate(selected):
        for ch in (0,1):
            ax=axes[ch,j]
            ax.plot(t,y[i,ch],lw=1.9,label="True",color=f"C{ch}")
            ax.plot(t,p[i,ch],lw=1.9,label="TCN",color=f"C{ch}",ls="--")
            ax.set(title=f"{sources[ch]} | FDG +{metadata[i]['delta_min']:g} min",
                   xlabel="Time (min)",ylabel=f"Activity ({unit})")
            ax.grid(alpha=.2)
    fig.suptitle(f"Per-tracer reference versus TCN prediction: {kind}",fontsize=14)
    _legend_below(fig,axes,ncol=2)
    path=out/"component_predictions.png";_save(fig,path)
    images["components"]=str(path)

    # Residual TACs, aligned y-range across injection-offset samples.
    residual=p-y
    limit=float(max(np.percentile(np.abs(residual),99.5),1e-8))*1.15
    fig,axes=plt.subplots(2,len(selected),figsize=(5.7*len(selected),6.6),squeeze=False)
    for j,i in enumerate(selected):
        for ch in (0,1):
            ax=axes[ch,j]
            ax.axhline(0,color="0.35",lw=.9,ls=":")
            ax.plot(t,residual[i,ch],color=f"C{ch}",lw=1.25,label=sources[ch]+" residual")
            ax.set_ylim(-limit,limit)
            ax.set(title=f"{sources[ch]} | FDG +{metadata[i]['delta_min']:g} min",
                   xlabel="Time (min)",ylabel=f"TCN - truth ({unit})")
            ax.grid(alpha=.2)
    fig.suptitle("TCN separation residuals (common y-scale)",fontsize=14)
    _legend_below(fig,axes,ncol=2)
    path=out/"component_residuals.png";_save(fig,path)
    images["residuals"]=str(path)

    # Predicted-versus-true: downsample scatter for large test sets.
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    rng=np.random.default_rng(123)
    for ch,ax in enumerate(axes):
        true=y[:,ch,:].ravel()
        pred=p[:,ch,:].ravel()
        choose=rng.choice(len(true),size=min(len(true),10000),replace=False)
        ax.scatter(true[choose],pred[choose],s=5,alpha=.12,color=f"C{ch}")
        low=float(min(np.min(true),np.min(pred)))
        high=float(max(np.max(true),np.max(pred)))
        ax.plot([low,high],[low,high],ls="--",color="black",lw=1,label="Identity")
        ax.set(xlabel=f"True ({unit})",ylabel=f"TCN ({unit})",title=sources[ch])
        ax.grid(alpha=.2)
    fig.suptitle("Framewise predicted versus true activity",fontsize=14)
    _legend_below(fig,axes,ncol=1)
    path=out/"prediction_parity.png";_save(fig,path)
    images["parity"]=str(path)

    # Error by frame, across all test examples.
    fig,axes=plt.subplots(1,2,figsize=(12.2,4.7))
    for ch,ax in enumerate(axes):
        rmse=np.sqrt(np.mean(residual[:,ch,:]**2,axis=0))
        mae=np.mean(np.abs(residual[:,ch,:]),axis=0)
        ax.plot(t,rmse,lw=1.7,label="RMSE")
        ax.plot(t,mae,lw=1.5,ls="--",label="MAE")
        ax.set(title=sources[ch],xlabel="Time from PBR28 injection (min)",
               ylabel=f"Error ({unit})")
        ax.grid(alpha=.2)
    fig.suptitle("TCN separation error over scan time",fontsize=14)
    _legend_below(fig,axes,ncol=2)
    path=out/"error_over_time.png";_save(fig,path)
    images["error_over_time"]=str(path)

    # Error distributions and per-offset median errors.
    fig,axes=plt.subplots(1,2,figsize=(12.2,4.8))
    for ch,ax in enumerate(axes):
        rr=residual[:,ch,:].ravel()
        lo,hi=np.quantile(rr,[.005,.995])
        ax.hist(rr,bins=70,range=(float(lo),float(hi)),alpha=.82,color=f"C{ch}")
        ax.axvline(0,color="black",ls="--",lw=1)
        ax.set(title=sources[ch],xlabel=f"TCN - truth ({unit})",ylabel="Frame count")
        ax.grid(alpha=.16)
    fig.suptitle("TCN residual distributions (0.5–99.5 percentile range)",fontsize=14)
    fig.subplots_adjust(top=.87,wspace=.3)
    path=out/"error_histograms.png";_save(fig,path)
    images["error_histograms"]=str(path)

    delay=np.array([float(v["delta_min"]) for v in metadata])
    offsets=np.unique(delay)
    fig,ax=plt.subplots(figsize=(8.6,4.7))
    for ch,name in enumerate(sources):
        vals=[float(np.sqrt(np.mean(residual[delay==d,ch,:]**2))) for d in offsets]
        ax.plot(offsets,vals,"o-",lw=1.7,label=name)
    ax.set(title="Test RMSE by FDG injection offset",
           xlabel="FDG injection offset (min)",ylabel=f"RMSE ({unit})")
    ax.grid(alpha=.2)
    _legend_below(fig,[ax],ncol=2)
    path=out/"error_by_offset.png";_save(fig,path)
    images["error_by_offset"]=str(path)

    # Percent errors normalized to each tracer's true RMS activity.
    # Avoid dividing by individual near-zero pre-injection frame values.
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    percent_by_offset={}
    reference_rms=np.sqrt(np.mean(y**2,axis=(0,2)))
    for ch,ax in enumerate(axes):
        divisor=max(float(reference_rms[ch]),1e-12)
        nrmse_pct=100*np.sqrt(np.mean(residual[:,ch,:]**2,axis=0))/divisor
        nmae_pct=100*np.mean(np.abs(residual[:,ch,:]),axis=0)/divisor
        ax.plot(t,nrmse_pct,lw=1.8,label="RMSE / reference RMS")
        ax.plot(t,nmae_pct,lw=1.5,ls="--",label="MAE / reference RMS")
        ax.set(title=sources[ch],xlabel="Time from PBR28 injection (min)",
               ylabel="Normalized error (%)")
        ax.grid(alpha=.2)
        percent_by_offset[sources[ch]]={
            str(float(d)):float(100*np.sqrt(np.mean(residual[delay==d,ch,:]**2))/
                         max(float(np.sqrt(np.mean(y[delay==d,ch,:]**2))),1e-12))
            for d in offsets}
    fig.suptitle("Percent error versus time (per-tracer global RMS normalization)",fontsize=14)
    _legend_below(fig,axes,ncol=2)
    path=out/"percent_error_over_time.png";_save(fig,path)
    images["percent_error_over_time"]=str(path)
    import csv
    with (out/"percent_error_by_offset.csv").open("w",newline="") as fp:
        writer=csv.writer(fp)
        writer.writerow(["tracer","fdg_offset_min","percent_rmse"])
        for tracer,offset_map in percent_by_offset.items():
            for key,val in offset_map.items():
                writer.writerow([tracer,key,val])

    # Mixture consistency.
    fig,axes=plt.subplots(1,len(selected),figsize=(5.7*len(selected),4.6),
                          squeeze=False)
    axes=axes.ravel()
    for ax,i in zip(axes,selected):
        ax.scatter(t,x[i,0],color="0.55",s=10,label="Measured mixture")
        ax.plot(t,c[i,0],color="black",label="Clean mixture")
        ax.plot(t,p[i].sum(axis=0),ls="--",color="C3",label="Sum of TCN outputs")
        ax.set(title=f"FDG +{metadata[i]['delta_min']:g} min",
               xlabel="Time (min)",ylabel=f"Activity ({unit})")
        ax.grid(alpha=.2)
    fig.suptitle("Physical mixture consistency",fontsize=14)
    _legend_below(fig,axes,ncol=3)
    path=out/"mixture_consistency.png";_save(fig,path)
    images["mixture_consistency"]=str(path)

    if training_history is not None and Path(training_history).exists():
        records=json.loads(Path(training_history).read_text())
        if records:
            fig,ax=plt.subplots(figsize=(8.7,4.8))
            epochs=[r["epoch"] for r in records]
            ax.plot(epochs,[r["train"]["loss"] for r in records],label="Train")
            ax.plot(epochs,[r["val"]["loss"] for r in records],label="Validation")
            ax.set(title="TCN training and validation loss",
                   xlabel="Epoch",ylabel="Normalized MSE-based loss")
            ax.grid(alpha=.2)
            _legend_below(fig,[ax],ncol=2)
            path=out/"training_curves.png";_save(fig,path)
            images["training_curves"]=str(path)
    return images
