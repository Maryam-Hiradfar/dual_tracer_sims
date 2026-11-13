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
    # Favor linear mappings with proven BQML units. A per-frame
    # RealWorldValueMappingSequence may be present but not have a linear
    # slope/intercept (e.g. LUT-only). Its presence must NOT mask a valid
    # shared PixelValueTransformationSequence.
    def candidates(name):
        per = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
        shared = getattr(ds, "SharedFunctionalGroupsSequence", None)
        for source, holder in (
            ("per-frame", per[i] if per is not None and len(per) > i else None),
            ("shared", shared[0] if shared else None),
            ("top-level", ds),
        ):
            if holder is None:
                continue
            seq = getattr(holder, name, None)
            if seq:
                for item in seq:
                    yield source, item

    slope = intercept = None
    rw_diagnostics = []
    for source, rw in candidates("RealWorldValueMappingSequence"):
        unit_seq = getattr(rw, "MeasurementUnitsCodeSequence", None)
        unit = str(getattr(unit_seq[0], "CodeValue", "")) if unit_seq else ""
        scheme = str(getattr(unit_seq[0], "CodingSchemeDesignator", "")) if unit_seq else ""
        a = getattr(rw, "RealWorldValueSlope", None)
        b = getattr(rw, "RealWorldValueIntercept", None)
        rw_diagnostics.append((source, unit, a is not None, b is not None))
        if a is None or b is None:
            continue
        normalized = unit.lower().replace(" ", "").replace("{", "").replace("}", "")
        if normalized not in ("bq/ml", "bq.ml-1", "bqml"):
            # The header says Units=BQML; ignore mappings to SUV and other units.
            # Do not mistake these for the quantitative PET concentration.
            continue
        slope, intercept = a, b
        break
    if slope is None:
        for source, transform in candidates("PixelValueTransformationSequence"):
            a = getattr(transform, "RescaleSlope", None)
            b = getattr(transform, "RescaleIntercept", None)
            if a is not None and b is not None:
                slope, intercept = a, b
                break
    if slope is None:
        slope = getattr(ds, "RescaleSlope", None)
        intercept = getattr(ds, "RescaleIntercept", None)
    # Legacy PET multiframe frequently stores only the first slice origin
    # in ImagePositionPatient, plus one GridFrameOffsetVector entry per slice.
    # This vector may specify offsets from the first position (relative form)
    # or absolute patient-z positions (absolute form). Only support the
    # unambiguous RELATIVE form; absolute variants require manual validation.
    if pos is not None and ori is not None and getattr(ds, "GridFrameOffsetVector", None) is not None:
        offsets = np.asarray(ds.GridFrameOffsetVector, dtype=float)
        nf = int(getattr(ds, "NumberOfFrames", 1))
        if len(offsets) != nf:
            raise ValueError("GridFrameOffsetVector length must match NumberOfFrames")
        if not np.isfinite(offsets).all() or not np.allclose(offsets[0], 0, atol=1e-4):
            raise ValueError("Unsupported GridFrameOffsetVector: expected relative offsets "
                             "starting at zero; inspect original DICOM orientation")
        original_pos = getattr(ds, "ImagePositionPatient", None)
        explicit_per_plane = field(ds, i, "PlanePositionSequence", "ImagePositionPatient")
        if explicit_per_plane is None and original_pos is not None:
            orientation = np.asarray(ori, float)
            if orientation.shape != (6,):
                raise ValueError("Invalid ImageOrientationPatient for grid offsets")
            direction = np.cross(orientation[:3], orientation[3:])
            if not np.isclose(np.linalg.norm(direction), 1, atol=1e-3):
                raise ValueError("Invalid image orientation for grid offsets")
            pos = np.asarray(original_pos, float) + offsets[i] * direction
    missing = [name for name, value in (
        ("ImageOrientationPatient", ori),
        ("ImagePositionPatient / GridFrameOffsetVector", pos),
        ("PixelSpacing", space),
        ("RescaleSlope / RealWorldValueSlope", slope),
        ("RescaleIntercept / RealWorldValueIntercept", intercept),
    ) if value is None]
    if missing:
        raise ValueError("Multiframe PET missing: " + ", ".join(missing) +
                         ". Run python -m sim.dicom_dynamic_pet --dicom-dir PATH --inspect")
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
        offsets = getattr(d, "GridFrameOffsetVector", None)
        print(" Classic geometry tags: "
              f"orientation={bool(getattr(d,'ImageOrientationPatient',None))}, "
              f"position={bool(getattr(d,'ImagePositionPatient',None))}, "
              f"pixel_spacing={bool(getattr(d,'PixelSpacing',None))}, "
              f"rescale_slope={getattr(d,'RescaleSlope',None)}, "
              f"rescale_intercept={getattr(d,'RescaleIntercept',None)}")
        if offsets is not None:
            print(f" GridFrameOffsetVector: count={len(offsets)}, "
                  f"first={list(offsets[:min(3,len(offsets))])}, "
                  f"last={list(offsets[-min(3,len(offsets)):])}")
        # Display only numeric calibration/units, never identifying fields.
        for scope,i in (("shared",0),("first",0),("middle",int(getattr(d,"NumberOfFrames",1))//2)):
            if scope=="shared":
                holder=sh[0] if sh else None
            else:
                holder=per[i] if per and len(per)>i else None
            if holder is None: continue
            pv=getattr(holder,"PixelValueTransformationSequence",None)
            rw=getattr(holder,"RealWorldValueMappingSequence",None)
            if pv:
                item=pv[0]
                print(f" {scope} pixel scaling: slope={getattr(item,'RescaleSlope',None)}, "
                      f"intercept={getattr(item,'RescaleIntercept',None)}")
            if rw:
                for j,item in enumerate(list(rw)[:3]):
                    us=getattr(item,"MeasurementUnitsCodeSequence",None)
                    unit=getattr(us[0],"CodeValue",None) if us else None
                    print(f" {scope} RWVM {j}: unit_code={unit!r}, "
                          f"slope={getattr(item,'RealWorldValueSlope',None)}, "
                          f"intercept={getattr(item,'RealWorldValueIntercept',None)}, "
                          f"LUT={bool(getattr(item,'RealWorldValueLUTData',None))}")
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
    import tempfile
    import os
    # Disk-backed intermediate avoids materializing a potentially 30+ GB 4D
    # volume in RAM. Saving uncompressed .nii enables memory-mapped frame reads.
    out=Path(out_nifti)
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.suffix==".gz":
        raise ValueError("Large dynamic PET conversion requires an uncompressed .nii "
                         "output for memory-mapped processing; set --converted-nifti "
                         "to an output path ending in .nii")
    fd,tmp=tempfile.mkstemp(prefix="enhanced_pet_",suffix=".dat",dir=out.parent)
    os.close(fd)
    array=np.memmap(tmp,dtype=np.float32,mode="w+",shape=(rows,cols,nslices,len(keys)))
    index={(rec["path"],rec["i"]):(t,z,rec)
           for t,frame in enumerate(slices) for z,rec in enumerate(frame)}
    try:
        for path in paths:
            d=pydicom.dcmread(str(path))
            pixel=np.asarray(d.pixel_array)
            if pixel.ndim!=3 or pixel.shape[1:]!=(rows,cols):
                raise ValueError("Unexpected decoded multiframe pixel array shape")
            for i in range(pixel.shape[0]):
                t,z,rec=index[(path,i)]
                array[:,:,z,t]=np.asarray(pixel[i],np.float32)*rec["slope"]+rec["intercept"]
        for ti in range(len(keys)):
            if not np.isfinite(array[...,ti]).all():
                raise ValueError(f"Nonfinite PET intensities in time frame {ti}")
        array.flush()
        img=nib.Nifti1Image(array,affine)
        img.header.set_xyzt_units("mm","sec")
        img.header.set_data_dtype(np.float32)
        nib.save(img,str(out))
    finally:
        del array
        Path(tmp).unlink(missing_ok=True)
    print(f"Enhanced PET converted: {len(keys)} dynamic frames, {nslices} slices, Bq/mL. "
          "Verify frame ordering and decay convention against source DICOM.")
    return out
