"""Enhanced/multiframe PET DICOM conversion with explicit spatial/temporal metadata.

Supports multiple enhanced instances (including one 3D volume per time point).
Rejects ambiguous temporal order, pixel scaling, and spatial geometry.
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import numpy as np


def fg(ds, i, name):
    per = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    shared = getattr(ds, "SharedFunctionalGroupsSequence", None)
    for holder in ((per[i] if per is not None and len(per)>i else None),
                   (shared[0] if shared else None)):
        seq = getattr(holder, name, None) if holder is not None else None
        if seq:
            return seq[0]
    return None


def field(ds, i, seq, name):
    obj = fg(ds, i, seq)
    return getattr(obj, name, None) if obj is not None else None


def frame_info(ds, i):
    def get(seq, fieldname):
        v = field(ds, i, seq, fieldname)
        return v if v is not None else getattr(ds, fieldname, None)
    ori = get("PlaneOrientationSequence", "ImageOrientationPatient")
    pos = get("PlanePositionSequence", "ImagePositionPatient")
    space = get("PixelMeasuresSequence", "PixelSpacing")
    slope = get("PixelValueTransformationSequence", "RescaleSlope")
    intercept = get("PixelValueTransformationSequence", "RescaleIntercept")
    rw = fg(ds, i, "RealWorldValueMappingSequence")
    if rw is None and getattr(ds, "RealWorldValueMappingSequence", None):
        rw = ds.RealWorldValueMappingSequence[0]
    if rw is not None:
        u = getattr(rw, "MeasurementUnitsCodeSequence", None)
        unit = str(getattr(u[0], "CodeValue", "")) if u else ""
        if unit.lower().replace(" ","") not in ("bq/ml","bq.ml-1"):
            raise ValueError(f"Real-world value map units {unit!r} not verified as Bq/mL")
        slope = getattr(rw,"RealWorldValueSlope",None)
        intercept = getattr(rw,"RealWorldValueIntercept",None)
    if any(v is None for v in (ori,pos,space,slope,intercept)):
        raise ValueError("Multiframe PET missing plane geometry or explicit BQML scaling")
    ref = field(ds,i,"FrameContentSequence","FrameReferenceTime")
    if ref is None:
        ref = getattr(ds,"FrameReferenceTime",None)
    ti = field(ds,i,"FrameContentSequence","TemporalPositionIndex")
    if ti is None:
        ti = getattr(ds,"TemporalPositionIdentifier",None)
    if ti is None:
        ti = getattr(ds,"AcquisitionNumber",None)
    if ref is not None:
        key = ("reference_ms",round(float(ref),3))
    elif ti is not None:
        key = ("temporal_index",int(ti))
    else:
        raise ValueError("No explicit time index for multiframe PET; cannot order dynamic frames")
    ori,pos,space = (np.asarray(v,dtype=float) for v in (ori,pos,space))
    slope,intercept = float(slope),float(intercept)
    if (ori.shape!=(6,) or pos.shape!=(3,) or space.shape!=(2,)
            or not all(np.isfinite(v).all() for v in (ori,pos,space))
            or np.min(space)<=0 or not np.isfinite(slope) or slope<=0
            or not np.isfinite(intercept)):
        raise ValueError("Invalid PET geometry or BQML scaling")
    return dict(key=key,ori=ori,pos=pos,space=space,slope=slope,intercept=intercept)


def inspect(directory, max_files=3):
    """Print non-identifying header information for troubleshooting."""
    import pydicom
    n=0
    for p in sorted(Path(directory).iterdir()):
        if not p.is_file(): continue
        try: d=pydicom.dcmread(str(p),stop_before_pixels=True)
        except Exception: continue
        if getattr(d,"Modality","")!="PT": continue
        n+=1
        per=getattr(d,"PerFrameFunctionalGroupsSequence",None)
        sh=getattr(d,"SharedFunctionalGroupsSequence",None)
        def seqs(x):
            return [el.keyword for el in x if el.VR=="SQ"] if x is not None else []
        print(f"PET #{n}: NumberOfFrames={getattr(d,'NumberOfFrames',1)}; "
              f"Rows={getattr(d,'Rows','?')}; Columns={getattr(d,'Columns','?')}; "
              f"Units={getattr(d,'Units','?')}; "
              f"FrameReferenceTime={getattr(d,'FrameReferenceTime',None)}; "
              f"AcquisitionNumber={getattr(d,'AcquisitionNumber',None)}")
        print(" Shared sequences:",seqs(sh[0] if sh else None))
        print(" Per-frame sequences:",seqs(per[0] if per else None))
        if per:
            for i in sorted({0,len(per)//2,len(per)-1}):
                fc=getattr(per[i],"FrameContentSequence",None)
                fc=fc[0] if fc else None
                print(f" Per-frame {i}: temporal index={getattr(fc,'TemporalPositionIndex',None)}, "
                      f"ref_time={getattr(fc,'FrameReferenceTime',None)}, "
                      f"plane_position={bool(getattr(per[i],'PlanePositionSequence',None))}")
        if n>=max_files: break
    print(f"Inspected {n} PET headers; patient identifiers not printed.")


def convert_multiframe(directory, out_nifti, *, timing_csv):
    import nibabel as nib
    import pydicom
    from sim.estimate_pet_noise import load_timing_csv
    records, paths, uids = [], [], set()
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file(): continue
        try: d=pydicom.dcmread(str(path),stop_before_pixels=True)
        except Exception: continue
        if getattr(d,"Modality","")!="PT": continue
        nf=int(getattr(d,"NumberOfFrames",1))
        if nf<=1: raise ValueError("Mixed classic and enhanced PET; isolate one series")
        if str(getattr(d,"Units","")).upper()!="BQML":
            raise ValueError("Multiframe PET not explicitly BQML; refuse uncertain scaling")
        uid=str(getattr(d,"SeriesInstanceUID",""))
        if not uid: raise ValueError("Missing SeriesInstanceUID")
        uids.add(uid)
        paths.append(path)
        for i in range(nf):
            rec=frame_info(d,i)
            rec.update(path=path,i=i,rows=int(d.Rows),cols=int(d.Columns))
            records.append(rec)
    if len(uids)!=1 or not records:
        raise ValueError("Expected exactly one enhanced PET series in the directory")
    groups=defaultdict(list)
    for r in records: groups[r["key"]].append(r)
    keys=sorted(groups,key=lambda k:k[1])
    if len(keys)<2 or len({k[0] for k in keys})!=1:
        raise ValueError("Cannot identify at least two unique, consistently indexed time frames")
    times=load_timing_csv(timing_csv,len(keys))
    ref=records[0]
    ori,space=ref["ori"],ref["space"]
    direction=np.cross(ori[:3],ori[3:])
    if not np.isclose(np.linalg.norm(direction),1,atol=1e-3):
        raise ValueError("Invalid image orientation")
    rows,cols=ref["rows"],ref["cols"]
    nslices=len(groups[keys[0]])
    if nslices<2: raise ValueError("Expected 3D PET volume with multiple slices/time")
    slices=[]; zz0=None
    for k in keys:
        items=groups[k]
        if len(items)!=nslices: raise ValueError("Missing or duplicated spatial slices")
        for r in items:
            if (r["rows"],r["cols"])!=(rows,cols) or not np.allclose(r["ori"],ori,atol=1e-4) or not np.allclose(r["space"],space,atol=1e-4):
                raise ValueError("Geometry differs between frames")
        items=sorted(items,key=lambda r:float(r["pos"]@direction))
        z=np.array([r["pos"]@direction for r in items])
        if np.any(np.diff(z)<=.001) or not np.allclose(np.diff(z),np.median(np.diff(z)),atol=.2,rtol=.01):
            raise ValueError("Invalid or duplicated slice positions")
        if zz0 is None:
            zz0=z
            origin=items[0]["pos"]
        elif not np.allclose(z,zz0,atol=.2) or not np.allclose(items[0]["pos"],origin,atol=.2):
            raise ValueError("Frame spatial positions differ")
        slices.append(items)
    affine_lps=np.eye(4)
    affine_lps[:3,0]=ori[3:]*space[0]
    affine_lps[:3,1]=ori[:3]*space[1]
    affine_lps[:3,2]=direction*np.median(np.diff(zz0))
    affine_lps[:3,3]=origin
    affine=np.diag([-1.,-1.,1.,1.])@affine_lps
    array=np.empty((rows,cols,nslices,len(keys)),dtype=np.float32)
    index={(rec["path"],rec["i"]):(t,z,rec)
           for t,frame in enumerate(slices) for z,rec in enumerate(frame)}
    for path in paths:
        d=pydicom.dcmread(str(path))
        pixel=np.asarray(d.pixel_array)
        if pixel.ndim!=3 or pixel.shape[1:]!=(rows,cols):
            raise ValueError("Unexpected decoded multiframe pixel array shape")
        for i in range(pixel.shape[0]):
            t,z,rec=index[(path,i)]
            array[:,:,z,t]=np.asarray(pixel[i],np.float32)*rec["slope"]+rec["intercept"]
    if not np.isfinite(array).all(): raise ValueError("Nonfinite PET intensity values")
    out=Path(out_nifti);out.parent.mkdir(parents=True,exist_ok=True)
    img=nib.Nifti1Image(array,affine)
    img.header.set_xyzt_units("mm","sec")
    img.header.set_data_dtype(np.float32)
    nib.save(img,str(out))
    print(f"Enhanced PET converted: {len(keys)} dynamic frames, {nslices} slices, Bq/mL. "
          "Verify frame ordering and decay convention against source DICOM.")
    return out
