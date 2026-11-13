"""Common anatomical ROI atlas loader.

Supports either a 3D integer-label NIfTI or a directory containing one
3D binary ROI-mask NIfTI per region. Filenames are deterministic region names.
All outputs are in the reference image grid. Overlapping masks are rejected.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


def load_atlas(source, *, reference=None, allow_resample=False, output_dir=None):
    """Return (integer label image, id->name).

    For a directory, sorted filenames get IDs 1..N. ROIs are thresholded >0.5.
    Use exactly the same directory across commands for stable IDs.
    """
    import nibabel as nib
    from nibabel.processing import resample_from_to

    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir():
        paths = sorted(set(source.glob("*.nii")) | set(source.glob("*.nii.gz")))
        if not paths:
            raise ValueError(f"No .nii/.nii.gz masks found in {source}; only direct children are read")
    else:
        paths = [source]

    target = reference
    names = {}
    if source.is_dir():
        if target is None:
            target = nib.load(str(paths[0]))
        if len(target.shape) != 3:
            raise ValueError("Reference atlas/PET spatial grid must be 3D")
        labels = np.zeros(target.shape, dtype=np.int32)
        for i, path in enumerate(paths, start=1):
            im = nib.load(str(path))
            if len(im.shape) != 3:
                raise ValueError(f"ROI mask must be 3D: {path}")
            if im.shape != target.shape or not np.allclose(im.affine, target.affine, atol=1e-3):
                if not allow_resample:
                    raise ValueError(f"ROI {path.name} grid differs from reference; "
                                     "register masks first or pass --resample-atlas")
                im = resample_from_to(im, (target.shape, target.affine), order=0)
            a = np.asarray(im.dataobj, dtype=float)
            if not np.isfinite(a).all():
                raise ValueError(f"ROI has nonfinite voxels: {path}")
            mask = a > .5
            if not mask.any():
                raise ValueError(f"Empty ROI after thresholding: {path}")
            if np.any(labels[mask] != 0):
                raise ValueError(f"Overlapping masks detected at {path.name}. "
                                 "Use disjoint parcellation ROIs; do not silently overwrite")
            labels[mask] = i
            name = path.name
            for suffix in (".nii.gz", ".nii"):
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    break
            names[i] = name
    else:
        im = nib.load(str(source))
        if len(im.shape) != 3:
            raise ValueError("label atlas must be a 3D integer NIfTI")
        if target is not None and (im.shape != target.shape or
                                   not np.allclose(im.affine, target.affine, atol=1e-3)):
            if not allow_resample:
                raise ValueError("atlas differs from reference; register or pass --resample-atlas")
            im = resample_from_to(im, (target.shape, target.affine), order=0)
        target = im
        raw = np.asanyarray(im.dataobj)
        if not np.isfinite(raw).all() or not np.allclose(raw, np.rint(raw), atol=1e-3):
            raise ValueError("atlas must contain finite integer-valued labels")
        labels = np.rint(raw).astype(np.int32)
        if np.any(labels < 0):
            raise ValueError("negative atlas labels not supported")
        names = {int(v): str(int(v)) for v in np.unique(labels) if v > 0}

    hdr = target.header.copy()
    hdr.set_data_dtype(np.int32)
    label_image = nib.Nifti1Image(labels, target.affine, hdr)
    if output_dir is not None and source.is_dir():
        import csv
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        nib.save(label_image, str(out / "combined_roi_labels.nii.gz"))
        with (out / "combined_roi_labels.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label_id", "region_name"])
            writer.writerows(sorted(names.items()))
    return label_image, names
