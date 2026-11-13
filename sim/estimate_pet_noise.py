"""Estimate regional framewise voxel noise from registered 4D PET and atlas NIfTIs.

Preferred: independent equal-count split-half reconstructions; fallback:
robust local spatial residuals (confounded by anatomy and reconstruction).
All activity inputs must be Bq/mL. Run from repository root.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter


def robust_sigma(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size else np.nan


def normalized_gaussian(data, mask, sigma):
    w = np.asarray(mask, float)
    numerator = gaussian_filter(np.where(mask, data, 0), sigma, mode="nearest")
    denominator = gaussian_filter(w, sigma, mode="nearest")
    return np.divide(numerator, denominator, out=np.zeros_like(numerator),
                     where=denominator > 1e-6)


def estimate_frame(frame, roi, *, half_a=None, half_b=None,
                   smooth_sigma_vox=2.0, erode_vox=1, min_voxels=30):
    frame, roi = np.asarray(frame, float), np.asarray(roi, bool)
    if frame.shape != roi.shape or frame.ndim != 3:
        raise ValueError("frame and ROI must be matching 3D arrays")
    if erode_vox < 0 or smooth_sigma_vox <= 0:
        raise ValueError("invalid erosion or smoothing parameters")
    interior = binary_erosion(roi, iterations=erode_vox) if erode_vox else roi
    good = interior & np.isfinite(frame)
    if good.sum() < min_voxels:
        return dict(n_voxels=int(good.sum()), mean_bqml=np.nan,
                    sigma_bqml=np.nan, cv=np.nan)
    mean = float(np.mean(frame[good]))
    if half_a is not None or half_b is not None:
        if half_a is None or half_b is None:
            raise ValueError("paired mode needs both independent half-data images")
        if half_a.shape != frame.shape or half_b.shape != frame.shape:
            raise ValueError("split reconstructions must match full PET shape")
        # For independent half-count reconstructions A and B:
        # Var(full) = Var(A-B)/4, assuming linear count-noise scaling.
        sigma = robust_sigma((np.asarray(half_a) - np.asarray(half_b))[good]) / 2
    else:
        trend = normalized_gaussian(frame, roi & np.isfinite(frame), smooth_sigma_vox)
        sigma = robust_sigma((frame - trend)[good])
    return dict(n_voxels=int(good.sum()), mean_bqml=mean, sigma_bqml=sigma,
                cv=float(sigma / mean) if mean > 0 else np.nan)


def load_timing_csv(path, n_frames):
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or len(rows) != n_frames or not {"TimeStart", "TimeEnd"}.issubset(rows[0]):
        raise ValueError("timing CSV must have a TimeStart/TimeEnd row per 4D frame (seconds)")
    edges = np.array([[float(r["TimeStart"]), float(r["TimeEnd"])] for r in rows])
    if (not np.isfinite(edges).all() or np.any(edges[:, 1] <= edges[:, 0])
            or np.any(np.diff(edges[:, 0]) <= 0)
            or np.any(edges[1:, 0] < edges[:-1, 1] - 1e-4)):
        raise ValueError("invalid timing CSV")
    return edges


def estimate_noise(pet_path, atlas_path, timing_path, output_dir, *,
                   half_a_path=None, half_b_path=None, atlas_lut=None,
                   smooth_sigma_vox=2.0, erode_vox=1, min_voxels=30,
                   voxel_volume_ml=None, allow_resample_atlas=False):
    import nibabel as nib
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if bool(half_a_path) != bool(half_b_path):
        raise ValueError("supply both --half-a and --half-b or neither")
    im = nib.load(str(pet_path))
    if len(im.shape) != 4:
        raise ValueError("PET must be 4D with time as final dimension")
    times = load_timing_csv(timing_path, im.shape[-1])
    from sim.brain_atlas_io import load_atlas
    # Build a 3D spatial reference from the 4D PET scan.
    reference = nib.Nifti1Image(np.zeros(im.shape[:3], dtype=np.uint8), im.affine)
    atlas, names = load_atlas(
        atlas_path, reference=reference, allow_resample=allow_resample_atlas,
        output_dir=output_dir,
    )
    labels = np.asarray(atlas.dataobj, dtype=np.int32)
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if not ids.size:
        raise ValueError("no atlas labels")
    if atlas_lut:
        lut = pd.read_csv(atlas_lut)
        if not {"label_id", "region_name"}.issubset(lut):
            raise ValueError("atlas LUT must have label_id,region_name")
        names.update({int(row.label_id): str(row.region_name)
                      for row in lut.itertuples(index=False)})
    if voxel_volume_ml is None:
        voxel_volume_ml = abs(np.linalg.det(im.affine[:3, :3])) / 1000
    if not np.isfinite(voxel_volume_ml) or voxel_volume_ml <= 0:
        raise ValueError("invalid voxel volume")
    a = nib.load(str(half_a_path)) if half_a_path else None
    b = nib.load(str(half_b_path)) if half_b_path else None
    for half in (a, b):
        if half is not None and (half.shape != im.shape or
                                 not np.allclose(half.affine, im.affine, atol=1e-3)):
            raise ValueError("half images must match full PET grid and frames")
    method = "independent_half_data" if a else "spatial_residual_proxy"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fi in range(im.shape[-1]):
        frame = np.asarray(im.dataobj[..., fi], dtype=np.float32)
        fa = np.asarray(a.dataobj[..., fi], dtype=np.float32) if a else None
        fb = np.asarray(b.dataobj[..., fi], dtype=np.float32) if b else None
        start, end = times[fi]
        for rid in ids:
            result = estimate_frame(frame, labels == rid, half_a=fa, half_b=fb,
                                    smooth_sigma_vox=smooth_sigma_vox,
                                    erode_vox=erode_vox, min_voxels=min_voxels)
            c, cv, dt = result["mean_bqml"], result["cv"], end - start
            # Equivalent Poisson scale; NOT a physical scanner NEMA sensitivity.
            equiv = (1 / (cv**2 * c * voxel_volume_ml * dt)
                     if np.isfinite(cv) and cv > 0 and np.isfinite(c) and c > 0
                     else np.nan)
            rows.append(dict(region_id=int(rid), region_name=names.get(int(rid), str(rid)),
                             frame_index=fi, TimeStart=start, TimeEnd=end,
                             TimeMid=(start + end) / 2, duration_s=dt,
                             **result, equivalent_sensitivity_cps_per_bq=equiv,
                             method=method))
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "noise_by_region_frame.csv", index=False)
    summary = df.groupby(["region_id", "region_name"], as_index=False).agg(
        median_cv=("cv", "median"),
        median_equivalent_sensitivity_cps_per_bq=("equivalent_sensitivity_cps_per_bq", "median"),
        n_valid_frames=("cv", "count"))
    summary.to_csv(output_dir / "noise_region_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for rid, d in df.groupby("region_id"):
        ax.plot(d["TimeMid"] / 60, d["cv"] * 100,
                label=names.get(int(rid), str(rid)), lw=1.2)
    ax.set(title=f"Regional voxel CV: {method}", xlabel="Frame midpoint (min)",
           ylabel="Estimated CV (%)")
    ax.grid(alpha=.2)
    if len(ids) <= 15:
        ax.legend(fontsize=8, ncol=2)
    fig.savefig(output_dir / "noise_cv_by_region.png", dpi=170)
    plt.close(fig)
    print(f"Saved {len(df)} frame-region estimates to {output_dir} ({method})")
    print("Caution: spatial residuals mix noise with heterogeneity; split-half estimates "
          "require independent equal-count reconstructions. Equivalent sensitivity is a proxy.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--pet", type=Path, help="4D Bq/mL PET NIfTI")
    source.add_argument("--pet-dicom-dir", type=Path, help="Directory of classic single-frame dynamic PET DICOM slices")
    p.add_argument("--atlas", type=Path, required=True,
                   help="3D label NIfTI or directory of individual ROI mask NIfTIs")
    p.add_argument("--timing-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/brain_noise"))
    p.add_argument("--half-a", type=Path)
    p.add_argument("--half-b", type=Path)
    p.add_argument("--atlas-lut", type=Path)
    p.add_argument("--smooth-sigma-vox", type=float, default=2.0)
    p.add_argument("--erode-vox", type=int, default=1)
    p.add_argument("--min-voxels", type=int, default=30)
    p.add_argument("--voxel-volume-ml", type=float)
    p.add_argument("--resample-atlas", action="store_true")
    p.add_argument("--converted-nifti", type=Path,
                   help="Output path for validated 4D PET NIfTI; default inside output-dir")
    a = p.parse_args()
    pet_path = a.pet
    if a.pet_dicom_dir is not None:
        from sim.dicom_dynamic_pet import convert_dicom_directory
        pet_path = a.converted_nifti or (a.output_dir / "converted_pet_bqml_4d.nii.gz")
        convert_dicom_directory(a.pet_dicom_dir, pet_path, timing_csv=a.timing_csv)
    estimate_noise(pet_path, a.atlas, a.timing_csv, a.output_dir, half_a_path=a.half_a,
                   half_b_path=a.half_b, atlas_lut=a.atlas_lut,
                   smooth_sigma_vox=a.smooth_sigma_vox, erode_vox=a.erode_vox,
                   min_voxels=a.min_voxels, voxel_volume_ml=a.voxel_volume_ml,
                   allow_resample_atlas=a.resample_atlas)


if __name__ == "__main__":
    main()
