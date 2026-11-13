"""Convert classic single-frame dynamic PET DICOM files from one directory to 4D Bq/mL NIfTI.

Conservative loader: requires explicit frame identity, uniform geometry, BQML units,
and external timing CSV. Rejects enhanced multiframe and uncertain scaling.
No patient metadata written. Source DICOM stays local.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np


def frame_key(d):
    """Prefer a PET frame reference time, then explicit temporal position."""
    if getattr(d, "FrameReferenceTime", None) is not None:
        return ("FrameReferenceTime", round(float(d.FrameReferenceTime), 3))
    if getattr(d, "TemporalPositionIdentifier", None) is not None:
        return ("TemporalPositionIdentifier", int(d.TemporalPositionIdentifier))
    if getattr(d, "AcquisitionNumber", None) is not None:
        return ("AcquisitionNumber", int(d.AcquisitionNumber))
    raise ValueError("No reliable dynamic frame identifier: need FrameReferenceTime, "
                     "TemporalPositionIdentifier or AcquisitionNumber; cannot infer from file names")


def _read_header(path, pydicom):
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True)
    except Exception:
        return None


def scan_series(directory):
    """Inspect classic PET DICOM headers. Return ordered spatial/temporal records."""
    import pydicom
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    groups = defaultdict(list)
    ignored = 0
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        d = _read_header(path, pydicom)
        if d is None or getattr(d, "Modality", "") != "PT":
            ignored += 1
            continue
        if int(getattr(d, "NumberOfFrames", 1)) != 1:
            raise ValueError("Enhanced/multiframe PET DICOM detected; conversion is not "
                             "supported by this conservative classic-DICOM loader")
        if str(getattr(d, "Units", "")).upper() != "BQML":
            raise ValueError(f"PET units are {getattr(d, 'Units', None)!r}, not BQML. "
                             "Do not estimate Bq/mL noise from SUV/CNTS images")
        if not all(hasattr(d, k) for k in
                   ("ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing",
                    "Rows", "Columns", "RescaleSlope", "RescaleIntercept")):
            raise ValueError(f"Incomplete PET geometry or activity scaling in {path.name}")
        if hasattr(d, "RealWorldValueMappingSequence"):
            raise ValueError("RealWorldValueMappingSequence present: manual verification "
                             "of SUV/BQML scaling is required")
        key = frame_key(d)
        groups[key].append((path, d))
    if len(groups) < 2:
        raise ValueError(f"Need at least 2 dynamic frames; found {len(groups)}. "
                         f"Ignored {ignored} non-PET/non-DICOM files")
    if len({k[0] for k in groups}) != 1:
        raise ValueError("Inconsistent temporal identifier types across files")
    # A single directory may contain interleaved slices of multiple frame series.
    # FrameReferenceTime is expected to increase monotonically with acquisition time.
    keys = sorted(groups, key=lambda k: k[1])
    first = groups[keys[0]][0][1]
    orient = np.asarray(first.ImageOrientationPatient, float)
    if orient.shape != (6,):
        raise ValueError("Unexpected ImageOrientationPatient")
    along_col = orient[:3]
    along_row = orient[3:]
    normal = np.cross(along_col, along_row)
    if not np.isclose(np.linalg.norm(normal), 1, atol=1e-3):
        raise ValueError("Invalid DICOM orientation")
    row_spacing, col_spacing = np.asarray(first.PixelSpacing, float)
    if min(row_spacing, col_spacing) <= 0:
        raise ValueError("Invalid pixel spacing")
    rows, cols = int(first.Rows), int(first.Columns)
    frame_len = len(groups[keys[0]])
    if frame_len < 2:
        raise ValueError("A frame appears to contain only one slice")
    sorted_groups = []
    z_first = None
    for key in keys:
        items = groups[key]
        if len(items) != frame_len:
            raise ValueError(f"Missing or duplicated slices in frame {key}: "
                             f"expected {frame_len}, got {len(items)}")
        for path, d in items:
            if (int(d.Rows), int(d.Columns)) != (rows, cols):
                raise ValueError("Mismatched PET matrix sizes")
            if not np.allclose(np.asarray(d.ImageOrientationPatient, float), orient, atol=1e-4):
                raise ValueError("Mismatched orientation")
            if not np.allclose(np.asarray(d.PixelSpacing, float),
                               [row_spacing, col_spacing], atol=1e-4):
                raise ValueError("Mismatched pixel spacing")
            if str(d.Units).upper() != "BQML":
                raise ValueError("Mixed pixel units")
        items = sorted(items, key=lambda item:
                       float(np.dot(np.asarray(item[1].ImagePositionPatient, float), normal)))
        z = np.array([float(np.dot(np.asarray(d.ImagePositionPatient, float), normal))
                      for _, d in items])
        if np.any(np.diff(z) < 0.001):
            raise ValueError("Duplicate or non-monotonic slice locations")
        if not np.allclose(np.diff(z), np.median(np.diff(z)), atol=0.2, rtol=0.01):
            raise ValueError("Irregular slice geometry")
        if z_first is None:
            z_first = z
            origin = np.asarray(items[0][1].ImagePositionPatient, float)
        elif not np.allclose(z, z_first, atol=0.2):
            raise ValueError("Frame slice locations differ")
        sorted_groups.append(items)
    dz = float(np.median(np.diff(z_first)))
    # Image array indexes are (row, column, slice); construct LPS -> RAS affine.
    affine_lps = np.eye(4)
    affine_lps[:3, 0] = along_row * row_spacing
    affine_lps[:3, 1] = along_col * col_spacing
    affine_lps[:3, 2] = normal * dz
    affine_lps[:3, 3] = origin
    affine = np.diag([-1., -1., 1., 1.]) @ affine_lps
    return sorted_groups, affine, (rows, cols, frame_len), keys


def convert_dicom_directory(directory, out_nifti, *, timing_csv):
    """Convert into a single 4D NIfTI after validating external timing."""
    import pydicom
    import nibabel as nib
    from sim.estimate_pet_noise import load_timing_csv
    groups, affine, shape, keys = scan_series(directory)
    times = load_timing_csv(timing_csv, len(groups))
    if (keys[0][0] == "FrameReferenceTime" and
            np.any(np.diff([k[1] for k in keys]) <= 0)):
        raise ValueError("Nonmonotonic FrameReferenceTime")
    # External TimeStart/TimeEnd fixes duration; header keys determine order.
    out_nifti = Path(out_nifti)
    out_nifti.parent.mkdir(parents=True, exist_ok=True)
    volume = np.empty((*shape, len(groups)), dtype=np.float32)
    for t, items in enumerate(groups):
        for z, (path, header) in enumerate(items):
            d = pydicom.dcmread(str(path))
            values = np.asarray(d.pixel_array, dtype=np.float32)
            if values.shape != shape[:2]:
                raise ValueError("Decoded pixel shape differs from DICOM header")
            # Per-image rescale: DICOM stored value -> quantitative BQML.
            slope = float(d.RescaleSlope)
            intercept = float(d.RescaleIntercept)
            if not np.isfinite(slope) or slope <= 0 or not np.isfinite(intercept):
                raise ValueError("Invalid BQML pixel rescale")
            volume[:, :, z, t] = values * slope + intercept
        if not np.isfinite(volume[..., t]).all():
            raise ValueError("Nonfinite scaled PET values")
    img = nib.Nifti1Image(volume, affine)
    img.header.set_xyzt_units("mm", "sec")
    img.header.set_data_dtype(np.float32)
    nib.save(img, str(out_nifti))
    print(f"Converted {len(groups)} frames × {shape[2]} slices to {out_nifti}")
    print(f"Frame identifier: {keys[0][0]}, timing CSV rows: {len(times)}")
    print("Verify activity calibration, decay-correction convention, and alignment "
          "against an original DICOM frame before quantitative noise modeling.")
    return out_nifti


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dicom-dir", type=Path, required=True)
    p.add_argument("--timing-csv", type=Path, required=True)
    p.add_argument("--output-nifti", type=Path, required=True)
    a = p.parse_args()
    convert_dicom_directory(a.dicom_dir, a.output_nifti, timing_csv=a.timing_csv)


if __name__ == "__main__":
    main()
