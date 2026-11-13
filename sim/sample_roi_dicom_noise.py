"""Sample a few enhanced multiframe PET DICOM volumes for ONE ROI noise proxy.

Avoids reconstructing all time frames or writing a massive 4D NIfTI.
Requires classic Panorama-style enhanced PET: one 3D volume/time point,
702-ish 2D planes per multiframe DICOM. Retains strict BQML/grid checks.
No raw image data uploaded or committed. Spatial residuals are not pure noise.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def select_frames(timing_rows, requested_durations=(5.,30.,60.,300.)):
    durations = np.diff(timing_rows,axis=1).ravel()
    picks = []
    for duration in requested_durations:
        idx = int(np.argmin(abs(durations-duration)))
        if idx not in picks: picks.append(idx)
    return sorted(picks)


def sample_one_roi(dicom_dir, roi_mask, timing_csv, output_dir, *,
                   target_durations=(5.,30.,60.,300.), resample_mask=False,
                   smooth_sigma_vox=2., erode_vox=1):
    import pydicom
    import nibabel as nib
    import pandas as pd
    from sim.enhanced_pet import frame_info
    from sim.estimate_pet_noise import estimate_frame, load_timing_csv
    from nibabel.processing import resample_from_to

    directory=Path(dicom_dir)
    if not directory.is_dir(): raise FileNotFoundError(directory)
    entries=[]
    uids=set()
    for path in sorted(directory.iterdir()):
        if not path.is_file(): continue
        try: d=pydicom.dcmread(str(path),stop_before_pixels=True)
        except Exception: continue
        if str(getattr(d,"Modality",""))!="PT": continue
        if str(getattr(d,"Units","")).upper()!="BQML":
            raise ValueError("Only quantitative Units=BQML PET supported")
        count=int(getattr(d,"NumberOfFrames",1))
        if count<2: raise ValueError("Expected multiframe 3D volume per time point")
        uids.add(str(getattr(d,"SeriesInstanceUID","")))
        first=frame_info(d,0)
        last=frame_info(d,count-1)
        if first["key"]!=last["key"]:
            raise ValueError("Time changes WITHIN a DICOM object; unsupported for sampled 3D volume")
        entries.append((first["key"],path))
    if len(uids)!=1 or not entries:
        raise ValueError("Expected exactly one enhanced PET series")
    if len({key for key,_ in entries})!=len(entries):
        raise ValueError("Duplicate time identifiers across PET DICOM files")
    entries.sort(key=lambda item:item[0])
    if len({key[0] for key,_ in entries})!=1:
        raise ValueError("Mixed temporal indexing schemes")
    times=load_timing_csv(timing_csv,len(entries))
    chosen=select_frames(times,target_durations)
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for index in chosen:
        key,path=entries[index]
        d=pydicom.dcmread(str(path))
        px=np.asarray(d.pixel_array)
        count=int(d.NumberOfFrames)
        if px.shape!=(count,int(d.Rows),int(d.Columns)):
            raise ValueError("unexpected decoded pixel array shape")
        info=[frame_info(d,i) for i in range(count)]
        if any(r["key"]!=key for r in info):
            raise ValueError("inconsistent time grouping within DICOM")
        first=info[0]
        ori=first["ori"];space=first["space"]
        normal=np.cross(ori[:3],ori[3:])
        if not np.isclose(np.linalg.norm(normal),1,atol=1e-3):
            raise ValueError("invalid orientation")
        order=np.argsort([p["pos"]@normal for p in info])
        ordered=[info[i] for i in order]
        z=np.array([r["pos"]@normal for r in ordered])
        if len(z)<2 or np.any(np.diff(z)<=1e-3) or not np.allclose(
                np.diff(z),np.median(np.diff(z)),atol=.2,rtol=.01):
            raise ValueError("slice positions invalid")
        if any(not np.allclose(r["ori"],ori,atol=1e-4) or
               not np.allclose(r["space"],space,atol=1e-4) for r in ordered):
            raise ValueError("inconsistent geometry inside one DICOM volume")
        A=np.eye(4)
        A[:3,0]=ori[3:]*space[0]
        A[:3,1]=ori[:3]*space[1]
        A[:3,2]=normal*np.median(np.diff(z))
        A[:3,3]=ordered[0]["pos"]
        A=np.diag([-1.,-1.,1.,1.])@A
        shape=(int(d.Rows),int(d.Columns),count)
        mask_img=nib.load(str(roi_mask))
        if len(mask_img.shape)!=3:
            raise ValueError("ROI mask must be a 3D NIfTI")
        if mask_img.shape!=shape or not np.allclose(mask_img.affine,A,atol=1e-3):
            if not resample_mask:
                raise ValueError("ROI does not match PET grid. Register to PET first, "
                                 "or use --resample-mask if already registered in world coordinates")
            mask_img=resample_from_to(mask_img,(shape,A),order=0)
        mask=np.asarray(mask_img.dataobj,dtype=float)>.5
        if not mask.any(): raise ValueError("empty ROI after alignment")
        vox=np.asarray(px[order],dtype=np.float32)
        slope=np.asarray([r["slope"] for r in ordered],dtype=np.float32)
        intercept=np.asarray([r["intercept"] for r in ordered],dtype=np.float32)
        # DICOM pixel data are (slice,row,col); NIfTI is (row,col,slice).
        volume=np.moveaxis(vox*slope[:,None,None]+intercept[:,None,None],0,-1)
        stats=estimate_frame(volume,mask,smooth_sigma_vox=smooth_sigma_vox,
                             erode_vox=erode_vox)
        dt=float(times[index,1]-times[index,0])
        vol_ml=abs(np.linalg.det(A[:3,:3]))/1000
        c,cv=stats["mean_bqml"],stats["cv"]
        seff=float(1/(cv*cv*c*vol_ml*dt)) if np.isfinite(cv) and cv>0 and c>0 else np.nan
        rows.append(dict(frame_index=index,TimeStart=times[index,0],
                         TimeEnd=times[index,1],TimeMid=times[index].mean(),
                         temporal_identifier=str(key),effective_sensitivity_cps_per_bq=seff,
                         voxel_volume_ml=vol_ml,**stats))
        print(f"frame {index} (duration {dt:g}s): ROI mean={c:.2f} Bq/mL, CV={cv:.3g}")
    df=pd.DataFrame(rows)
    df.to_csv(output_dir/"single_roi_noise_samples.csv",index=False)
    valid=df["effective_sensitivity_cps_per_bq"].to_numpy(float)
    valid=valid[np.isfinite(valid)&(valid>0)]
    if len(valid):
        (output_dir/"single_roi_noise_summary.json").write_text(__import__("json").dumps({
            "median_equivalent_sensitivity_cps_per_bq":float(np.median(valid)),
            "source":"one ROI, local spatial-residual noise proxy; NOT direct detector sensitivity",
            "n_frames_sampled":len(df)},indent=2)+"\n")
    return df


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pet-dicom-dir",type=Path,required=True)
    p.add_argument("--roi-mask",type=Path,required=True,
                   help="ONE 3D binary brain ROI mask, preferably white matter")
    p.add_argument("--timing-csv",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,default=Path("results/single_roi_noise"))
    p.add_argument("--target-durations-sec",type=float,nargs="+",default=[5,30,60,300])
    p.add_argument("--resample-mask",action="store_true")
    p.add_argument("--smooth-sigma-vox",type=float,default=2.)
    p.add_argument("--erode-vox",type=int,default=1)
    a=p.parse_args()
    sample_one_roi(a.pet_dicom_dir,a.roi_mask,a.timing_csv,a.output_dir,
                   target_durations=a.target_durations_sec,resample_mask=a.resample_mask,
                   smooth_sigma_vox=a.smooth_sigma_vox,erode_vox=a.erode_vox)


if __name__=="__main__":
    main()
