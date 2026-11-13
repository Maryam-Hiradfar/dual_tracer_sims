"""Populate registered 3D brain-label NIfTI with FDG/PBR28 ROI TACs.

Inputs are decay-corrected Bq/mL, TimeMid in seconds. Synthesized PET is
not reconstructed real dual-tracer data. No source data are committed.
"""
from __future__ import annotations
import argparse, csv, json, tempfile
from pathlib import Path
import numpy as np
import pandas as pd


def table(path):
    path = Path(path)
    d = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    if "TimeMid" not in d or np.any(np.diff(d.TimeMid.to_numpy(float)) <= 0):
        raise ValueError("TAC needs increasing TimeMid values (seconds)")
    return d


def read_edges(path, pbr, duration):
    if path:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        st = np.array([float(r["TimeStart"]) for r in rows]) / 60
        en = np.array([float(r["TimeEnd"]) for r in rows]) / 60
    elif {"TimeStart", "TimeEnd"}.issubset(pbr.columns):
        st = pbr.TimeStart.to_numpy(float) / 60
        en = pbr.TimeEnd.to_numpy(float) / 60
    else:
        raise ValueError("PBR TAC needs TimeStart/TimeEnd or pass --frame-edges-csv")
    if not np.isclose(st[0], 0) or np.any(en <= st) or not np.allclose(st[1:], en[:-1], atol=1e-5):
        raise ValueError("output frame edges must be contiguous and start at 0")
    if duration <= 0 or duration > en[-1] + 1e-8:
        raise ValueError("duration exceeds schedule")
    return np.r_[st[st < duration - 1e-8], duration]


def concentration(t, midpoint, measured, injection):
    x = np.asarray(t) - injection
    c = np.interp(np.maximum(x, 0), midpoint, measured)
    first = midpoint[0]
    if first > 0:
        c = np.where((x > 0) & (x < first), measured[0] * x / first, c)
    return np.where(x > 0, c, 0.)


def physical_frames(mid, measured, edges, injection, half_life):
    if not np.isfinite(measured).all() or np.any(measured < 0):
        raise ValueError("TAC must contain nonnegative finite Bq/mL")
    out = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= injection:
            out.append(0.); continue
        lo = max(left, injection)
        t = np.linspace(lo, right, max(2, int(np.ceil((right-lo)*60))+1))
        curve = concentration(t, mid, measured, injection)
        decay = np.exp(-np.log(2)*(t-injection)/half_life)
        out.append(float(np.trapezoid(curve*decay, t)/(right-left)))
    return np.array(out)


def make_curves(fdg, pbr, mapping, edges, offset):
    fdg_t = fdg.TimeMid.to_numpy(float)/60
    pbr_t = pbr.TimeMid.to_numpy(float)/60
    def coverage(d, t):
        return float(d.TimeEnd.iloc[-1])/60 if "TimeEnd" in d else t[-1]
    if edges[-1] > coverage(pbr,pbr_t)+1e-8 or edges[-1]-offset > coverage(fdg,fdg_t)+1e-8:
        raise ValueError("exceeds measured TAC coverage; no extrapolation allowed")
    records = []
    for row in mapping.itertuples(index=False):
        if row.fdg_column not in fdg or row.pbr_column not in pbr:
            raise ValueError(f"missing ROI TAC column for {row.region_name}")
        a = physical_frames(fdg_t, fdg[row.fdg_column].to_numpy(float), edges, offset, 109.8)
        b = physical_frames(pbr_t, pbr[row.pbr_column].to_numpy(float), edges, 0, 20.4)
        for f in range(len(a)):
            records.append(dict(region_id=int(row.label_id), region_name=row.region_name,
                                frame_index=f, TimeStart=edges[f]*60,
                                TimeEnd=edges[f+1]*60, TimeMid=(edges[f]+edges[f+1])*30,
                                fdg_phys_bqml=a[f], pbr_phys_bqml=b[f],
                                mixture_clean_bqml=a[f]+b[f]))
    return pd.DataFrame(records)


def noise_lookup(path, ids):
    d = pd.read_csv(path)
    key = "median_equivalent_sensitivity_cps_per_bq"
    if key in d:
        values = d.set_index("region_id")[key]
    elif "equivalent_sensitivity_cps_per_bq" in d:
        values = d.groupby("region_id")["equivalent_sensitivity_cps_per_bq"].median()
    else:
        raise ValueError("wrong noise-calibration CSV columns")
    result = {int(i): float(values.loc[i]) for i in ids if i in values.index}
    if set(result) != set(ids) or any(not np.isfinite(v) or v <= 0 for v in result.values()):
        raise ValueError("noise calibration missing valid regions")
    return result


def populate(atlas_path, fdg_path, pbr_path, mapping_path, output_dir, *,
             duration=60., offset=5., frame_csv=None, previews=(5.,20.,50.),
             save_4d=False, noise_csv=None, noise_sensitivity=None, seed=48):
    import nibabel as nib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from numpy.lib.format import open_memmap
    from sim.brain_atlas_io import load_atlas
    atlas, atlas_names = load_atlas(atlas_path, output_dir=output_dir)
    labels = np.asarray(atlas.dataobj, dtype=np.int32)
    fdg, pbr = table(fdg_path), table(pbr_path)
    mapping = pd.read_csv(mapping_path)
    if not {"label_id","region_name","fdg_column","pbr_column"}.issubset(mapping.columns):
        raise ValueError("mapping needs label_id,region_name,fdg_column,pbr_column")
    if mapping.label_id.duplicated().any() or mapping[["fdg_column","pbr_column"]].isna().any().any():
        raise ValueError("mapping IDs must be unique and columns not blank")
    if not set(mapping.label_id.astype(int)).issubset(set(np.unique(labels))):
        raise ValueError("mapping contains label IDs not present in the atlas")
    edges = read_edges(frame_csv, pbr, duration)
    curves = make_curves(fdg, pbr, mapping, edges, offset)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(output_dir/"phantom_roi_tacs.csv",index=False)
    grouped = {int(i):d.sort_values("frame_index") for i,d in curves.groupby("region_id")}
    if noise_csv and noise_sensitivity:
        raise ValueError("choose noise CSV or fixed sensitivity, not both")
    sens = (noise_lookup(noise_csv, set(grouped)) if noise_csv else
            {i:noise_sensitivity for i in grouped} if noise_sensitivity else {})
    if sens and any(v <= 0 or not np.isfinite(v) for v in sens.values()):
        raise ValueError("invalid noise sensitivity")
    voxel_ml = abs(np.linalg.det(atlas.affine[:3,:3]))/1000
    if voxel_ml <= 0:
        raise ValueError("invalid voxel volume")
    rng = np.random.default_rng(seed)
    mids = (edges[:-1]+edges[1:])/2
    idx = sorted(set(int(np.argmin(abs(mids-t))) for t in previews))
    names = ["fdg","pbr","mixture"] + (["noisy_mixture"] if sens else [])
    def build(i):
        maps={name:np.zeros(labels.shape,np.float32) for name in names}
        for rid, row in grouped.items():
            region = labels==rid
            maps["fdg"][region]=row.fdg_phys_bqml.iloc[i]
            maps["pbr"][region]=row.pbr_phys_bqml.iloc[i]
        maps["mixture"]=maps["fdg"]+maps["pbr"]
        if sens:
            duration_s=(edges[i+1]-edges[i])*60
            for rid,s in sens.items():
                m=labels==rid
                alpha=voxel_ml*s*duration_s
                maps["noisy_mixture"][m]=rng.poisson(maps["mixture"][m]*alpha)/alpha
        return maps
    snapshots={}
    if save_4d:
        with tempfile.TemporaryDirectory(dir=output_dir,prefix="phantom_tmp_") as tmp:
            mm={name:open_memmap(Path(tmp)/f"{name}.npy",mode="w+",dtype="float32",
                                 shape=(*labels.shape,len(mids))) for name in names}
            for i in range(len(mids)):
                maps=build(i)
                for name in names: mm[name][...,i]=maps[name]
                if i in idx: snapshots[i]=maps
            for name in names:
                mm[name].flush()
                header=atlas.header.copy()
                header.set_data_dtype(np.float32)
                nib.save(nib.Nifti1Image(mm[name],atlas.affine,header),
                         str(output_dir/f"{name}_4d.nii.gz"))
    else:
        for i in idx: snapshots[i]=build(i)
    fig,axs=plt.subplots(len(idx),len(names),figsize=(4*len(names),3.5*len(idx)),
                         squeeze=False,constrained_layout=True)
    z=labels.shape[2]//2
    vmax=max(float(np.max(snapshots[i]["mixture"])) for i in idx)
    for row,i in enumerate(idx):
        for col,name in enumerate(names):
            axs[row,col].imshow(np.rot90(snapshots[i][name][:,:,z]),cmap="inferno",
                                vmin=0,vmax=vmax)
            axs[row,col].set_title(f"{name} | {mids[i]:.1f} min")
            axs[row,col].axis("off")
    fig.suptitle("Synthetic ROI-populated FDG/PBR28 brain phantom (NOT real dual-tracer PET)")
    fig.savefig(output_dir/"phantom_preview.png",dpi=170)
    plt.close(fig)
    fig,axs=plt.subplots(min(6,len(grouped)),1,figsize=(11,min(6,len(grouped))*3),
                         squeeze=False,constrained_layout=True)
    for ax,(rid,d) in zip(axs[:,0],list(grouped.items())[:6]):
        for col,lab in [("fdg_phys_bqml","FDG"),("pbr_phys_bqml","PBR28"),
                        ("mixture_clean_bqml","Mixture")]:
            ax.plot(mids,d[col],label=lab)
        ax.set(title=str(d.region_name.iloc[0]),xlabel="Minutes",ylabel="Bq/mL")
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(output_dir/"phantom_roi_curves.png",dpi=170)
    plt.close(fig)
    (output_dir/"phantom_metadata.json").write_text(json.dumps(dict(
        units="Bq/mL",frame_edges_min=edges.tolist(),fdg_offset_min=offset,
        atlas=str(atlas_path),simulation="ROI-uniform anatomical label phantom",
        physically_decaying_components=True,noise="independent effective Poisson" if sens else "none",
        disclaimer="Not reconstruction noise; scanner physics and spatial noise correlations omitted",
        seed=seed),indent=2)+"\n")
    if set(np.unique(labels))-{0}-set(grouped):
        print("WARNING: unmapped atlas regions remain zero")
    print(f"Saved synthetic phantom to {output_dir}; use --save-4d for NIfTI volumes")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atlas",type=Path,required=True,
                   help="Integer atlas NIfTI or directory of ROI mask NIfTIs")
    p.add_argument("--fdg-rois",type=Path,required=True)
    p.add_argument("--pbr-rois",type=Path,required=True)
    p.add_argument("--mapping-csv",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,default=Path("results/brain_phantom"))
    p.add_argument("--duration-min",type=float,default=60)
    p.add_argument("--fdg-offset-min",type=float,default=5)
    p.add_argument("--frame-edges-csv",type=Path)
    p.add_argument("--preview-min",type=float,nargs="+",default=[5,20,50])
    p.add_argument("--noise-calibration",type=Path)
    p.add_argument("--noise-sensitivity",type=float)
    p.add_argument("--save-4d",action="store_true")
    p.add_argument("--seed",type=int,default=48)
    a=p.parse_args()
    populate(a.atlas,a.fdg_rois,a.pbr_rois,a.mapping_csv,a.output_dir,
             duration=a.duration_min,offset=a.fdg_offset_min,
             frame_csv=a.frame_edges_csv,previews=a.preview_min,
             noise_csv=a.noise_calibration,noise_sensitivity=a.noise_sensitivity,
             save_4d=a.save_4d,seed=a.seed)


if __name__=="__main__":
    main()
