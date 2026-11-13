"""Multipage PDF of EVERY synthetic TCN test case.

Groups test examples by FDG injection offset and renders up to four cases per
page. No additional model fitting; uses the trained checkpoint's predictions.
All legends are placed below the subplot grid, outside data areas.
"""
from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ml.tcn_plot_style import use_slide_plot_style

use_slide_plot_style()
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def export_all_test_cases_pdf(
    path, *, time_min, measured, clean, target, prediction, metadata,
    unit="Bq/mL", cases_per_page=4,
):
    """Write 1 PDF containing every example in the passed test split.

    Arrays: measured/clean (N,1,T); targets/predictions (N,2,T).
    Percentage error = 100*RMSE/RMS(true tracer), per case, per tracer;
    undefined if a tracer's true RMS is zero.
    """
    t=np.asarray(time_min,dtype=float)
    observed=np.asarray(measured,dtype=float)
    clean=np.asarray(clean,dtype=float)
    true=np.asarray(target,dtype=float)
    pred=np.asarray(prediction,dtype=float)
    if (true.ndim!=3 or true.shape[1]!=2 or true.shape!=pred.shape or
        observed.shape!=(true.shape[0],1,true.shape[-1]) or
        clean.shape!=observed.shape or t.shape!=(true.shape[-1],) or
        len(metadata)!=len(true) or len(true)==0):
        raise ValueError("Invalid test data shapes for PDF")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Nonfinite target or predicted activity")
    if cases_per_page<1 or cases_per_page>6:
        raise ValueError("cases_per_page must be 1 to 6")
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    # Stable sort: original order preserved within each offset.
    order=sorted(range(len(true)),
                 key=lambda i: (float(metadata[i].get("delta_min",0)),i))
    groups={}
    for idx in order:
        groups.setdefault(float(metadata[idx].get("delta_min",0)),[]).append(idx)
    n_total_pages=sum(ceil(len(indices)/cases_per_page) for indices in groups.values())
    ncols=2 if cases_per_page>1 else 1
    nrows=ceil(cases_per_page/ncols)
    page=0
    with PdfPages(path) as pdf:
        info=pdf.infodict()
        info["Title"]="Dual-tracer TCN: all held-out test cases"
        info["Subject"]="Per-case true versus predicted tracer TACs, grouped by injection offset"
        for offset,indices in groups.items():
            for begin in range(0,len(indices),cases_per_page):
                subset=indices[begin:begin+cases_per_page]
                page+=1
                fig,axes=plt.subplots(nrows,ncols,figsize=(17.5,5.0*nrows+1.65),
                                      squeeze=False)
                axes=axes.ravel()
                for ax,i in zip(axes,subset):
                    for ch,name,color in ((0,"FDG","C0"),(1,"PBR28","C1")):
                        ax.plot(t,true[i,ch],color=color,linewidth=3.0,
                                label=name+" truth")
                        ax.plot(t,pred[i,ch],color=color,linewidth=2.8,
                                linestyle="--",label=name+" TCN")
                    ax.plot(t,clean[i,0],color="black",lw=2.5,label="Clean mixture")
                    ax.scatter(t,observed[i,0],color=".50",s=22,alpha=.65,
                               label="Noisy mixture")
                    errors=[]
                    for ch,name in ((0,"FDG"),(1,"PBR28")):
                        y=true[i,ch]
                        delta=pred[i,ch]-y
                        rmse=float(np.sqrt(np.mean(delta**2)))
                        scale=float(np.sqrt(np.mean(y**2)))
                        pct=f"{100*rmse/scale:.1f}%" if scale>0 else "N/A"
                        errors.append(f"{name}: {rmse:.3g} {unit} ({pct})")
                    sample_id=metadata[i].get("sample_id",i)
                    ax.set_title(f"Test index {i} | sample {sample_id}\n"+
                                 "   |   ".join(errors),fontsize=12,fontweight="bold")
                    ax.set_xlabel("Time from PBR28 injection (min)")
                    ax.set_ylabel(f"Activity ({unit})")
                    ax.grid(alpha=.20)
                for ax in axes[len(subset):]:
                    ax.axis("off")
                fig.suptitle(
                    f"Dual-tracer TCN | FDG injection +{offset:g} min | "
                    f"{len(subset)} case(s) | page {page}/{n_total_pages}",
                    fontsize=19,fontweight="bold",y=.99)
                handles,labels=axes[0].get_legend_handles_labels()
                fig.legend(handles,labels,loc="lower center",
                           bbox_to_anchor=(.5,.013),ncol=6,frameon=False,prop={"size":13,"weight":"bold"})
                fig.tight_layout(rect=(0,.095,1,.925),pad=2.6)
                pdf.savefig(fig)
                plt.close(fig)
    return {"pdf_path":str(path),"test_cases":len(true),
            "pages":page,"offset_groups":{str(k):len(v) for k,v in groups.items()}}


def main():
    """Regenerate PDF directly from an existing evaluation without model inference."""
    import argparse
    import json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,required=True,
                   help="Dataset with frame_midpoints and test_metadata.json")
    p.add_argument("--evaluation-dir",type=Path,required=True,
                   help="Folder with test_predictions.npz")
    p.add_argument("--out",type=Path,
                   help="Defaults to evaluation-dir/all_test_cases.pdf")
    p.add_argument("--cases-per-page",type=int,default=4)
    a=p.parse_args()
    meta=json.loads((a.data_dir/"dataset_metadata.json").read_text())
    examples=json.loads((a.data_dir/"test_metadata.json").read_text())
    saved=np.load(a.evaluation_dir/"test_predictions.npz")
    result=export_all_test_cases_pdf(
        a.out or a.evaluation_dir/"all_test_cases.pdf",
        time_min=meta["frame_midpoints_min"],
        measured=saved["measured_tavg_n1_d1"],
        clean=saved["clean_tavg_n0_d1"],
        target=saved["target_phys_tavg_n0_d1"],
        prediction=saved["prediction_phys_tavg_n0_d1"],
        metadata=examples,
        unit=meta.get("units",{}).get("TAC","Bq/mL"),
        cases_per_page=a.cases_per_page)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
