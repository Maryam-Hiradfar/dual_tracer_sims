# Empirical brain ROI phantom and PET noise estimation (experimental)

**Status:** Prototype scripts. Requires validation on real NIfTI files and correct ROI mapping. No participant data should be pushed to GitHub.

Install dependencies in your local `.venv` (or the UMIP1 environment):

```bash
python -m pip install numpy scipy pandas matplotlib nibabel openpyxl pytest
python -m pytest tests/test_brain_phantom.py -v
```

## Atlas provided as a directory of individual ROI NIfTIs

All three commands now accept **either** a single 3D integer-label NIfTI **or**
a directory with one 3D ROI mask (`.nii`/`.nii.gz`) per brain region.
For example:

```text
brain_rois/
  Left_Cortex.nii.gz
  Left_WhiteMatter.nii.gz
  Right_Cortex.nii.gz
  Right_WhiteMatter.nii.gz
```

Replace `--atlas "/private/brain_labels.nii.gz"` in any command with
`--atlas "/private/brain_rois"`. Masks are assigned deterministic ascending
integer labels by sorted filename. The scripts write the derived
`combined_roi_labels.nii.gz` and `combined_roi_labels.csv` into the output
directory. The mapping generator also suggests matching TAC spreadsheet columns
using mask filenames. Review every row; any blanks must be completed or their
corresponding mask removed from the input set.

**Important:** Every mask must already be on the same voxel grid and affine.
For real PET noise estimation, the masks must also align with PET; use
`--resample-atlas` only if they are correctly registered in world coordinates
and require nearest-neighbor grid resampling. An MRI-native parcellation with
no PET-MRI registration cannot be made anatomically correct simply by
resampling. Overlapping mask voxels are rejected by default. If your ROI
directory has overlapping or nested masks (e.g. whole brain + cortex),
create a disjoint subset first. Keep the identical mask directory/filenames
for noise estimation, ROI mapping and phantom population, because numerical
label IDs are assigned from sorted files.

Example: estimate noise directly from PET DICOM and an ROI-mask directory:

```bash
python -m sim.estimate_pet_noise \
  --pet-dicom-dir "/private/FDG_DICOM" \
  --atlas "/private/brain_rois_in_PET_space" \
  --timing-csv "/private/FDG_timing.csv" \
  --output-dir results/FDG_noise
```

## Enhanced / multiframe Panorama PET DICOM

The DICOM converter now detects PET instances with `NumberOfFrames > 1`
and attempts conservative conversion using the shared/per-frame functional
groups. It supports multiframe objects when each decoded 2D plane has an
explicit 3D patient-space location, a verifiable temporal position or
reference time, and quantitative BQML rescaling. It does not assume that
the flattened multiframe dimension is time rather than slice.

Re-run the same noise-estimation command with `--pet-dicom-dir`, not `--pet`.
If a vendor-specific export is rejected, inspect the headers **without**
printing identifying metadata:

```bash
python -m sim.dicom_dynamic_pet \
  --dicom-dir "/path/to/WB PET Dynamic 3i10s_160730_401" \
  --inspect
```

This reports the frame count, image dimensions, activity units, selected
frame-reference tags and names of nested metadata sequences. Share that
text-only output if the converter cannot determine frame geometry or order.
The synthetic enhanced-PET regression test can be run via:

```bash
python -m pytest tests/test_enhanced_pet.py -v
```

**Scientific QA before noise estimation:** compare reconstructed frame
midpoints to `TimeStart/TimeEnd` CSV, check image orientation against a
known anatomical landmark, and compare one or more original DICOM pixel
values after quantitative scaling against the 4D NIfTI. Pixel intensities
might be scan-start decay-corrected; verify that convention independently.
Do not interpret the fitted equivalent Poisson sensitivity as a measured
scanner NEMA sensitivity.

## Dynamic PET DICOM in a single directory

The noise estimator accepts classic **single-frame PET DICOMs** directly with
`--pet-dicom-dir` (no pre-conversion necessary). Files can be extensionless;
the program reads DICOM headers, groups slices into frames using
`FrameReferenceTime`, `TemporalPositionIdentifier`, or `AcquisitionNumber`,
sorts slices by patient-space position, applies per-file
`RescaleSlope/RescaleIntercept` for DICOM `Units=BQML`, and saves a local
converted 4D NIfTI. It requires a **frame timing CSV** in seconds matching the
number of dynamic frames.

```bash
python -m pip install pydicom nibabel
python -m sim.estimate_pet_noise \
  --pet-dicom-dir "/private/FDG/DICOM" \
  --atlas "/private/brain_labels_in_PET_space.nii.gz" \
  --timing-csv "/private/FDG_timing.csv" \
  --output-dir results/FDG_brain_noise
```

Inspect `results/FDG_brain_noise/converted_pet_bqml_4d.nii.gz` with your
original DICOM viewer to verify frame order, orientation, scaling and
decay-correction convention before trusting noise estimates.

**Limitations:** Enhanced PET requires explicit per-frame plane positions and time grouping; unsupported variants are rejected rather than guessed. Some exports use a single
`AcquisitionNumber` for all frames or assign varying identifiers per slice.
If automatic temporal grouping fails, inspect header metadata before
converting. A mixed directory containing multiple reconstructions or PET
series should be separated first. The current loader uses temporal frame keys
and supplied timing as a consistency check on frame count; check actual
frame-reference timestamps against CSV midpoints where available.
Some compressed DICOM transfer syntaxes require a pixel decoding plugin
such as `pylibjpeg` or `gdcm`.

## 1. Estimate frame-dependent voxel noise from real PET

Supply a **4D Bq/mL PET NIfTI** with time in its last dimension, a **registered 3D integer-label atlas**, and a frame timing CSV containing `TimeStart,TimeEnd` in **seconds**, one row per dynamic frame.

```bash
python -m sim.estimate_pet_noise \
  --pet "/private/FDG_dynamic_4d.nii.gz" \
  --atlas "/private/brain_labels_in_FDG_space.nii.gz" \
  --timing-csv "/private/FDG_timing.csv" \
  --output-dir results/FDG_brain_noise \
  --smooth-sigma-vox 2 \
  --erode-vox 1
```

The default algorithm calculates robust local high-pass residual variation within eroded atlas regions, as an **approximate voxel noise proxy**. This method is confounded by tissue heterogeneity, resolution, reconstruction and motion. Strong gradients in small regions may compromise estimates. Outputs include `noise_by_region_frame.csv`, `noise_region_summary.csv` and `noise_cv_by_region.png`.

**Preferred, if available:** independent equal-count list-mode split-half reconstructions, with identical geometry, time frames, units and reconstruction settings:

```bash
python -m sim.estimate_pet_noise \
  --pet "/private/FDG_full_4d.nii.gz" \
  --half-a "/private/FDG_half_A_4d.nii.gz" \
  --half-b "/private/FDG_half_B_4d.nii.gz" \
  --atlas "/private/brain_labels_in_FDG_space.nii.gz" \
  --timing-csv "/private/FDG_timing.csv" \
  --output-dir results/FDG_noise_halves
```

Approximate full-count reconstructed voxel standard deviation:
`sigma_full = robust_sigma(half_A-half_B)/2`, assuming equal independent split noise and variance scaling inversely with counts.

The script also reports an **equivalent Poisson coefficient** under the idealized relation
`CV² ≈ 1/(C * V * S_eff * frame_duration_seconds)`. This empirical coefficient is **not** measured scanner sensitivity and may differ by tracer, timing, region, and reconstruction.

## 2. Populate an anatomical atlas with measured ROI TACs

Required inputs:

- A real, anatomically registered 3D **integer-valued labelmap NIfTI** (e.g. your Neurosegmenter or CerebrA parcellation).
- FDG and PBR28 Excel/CSV ROI TAC tables. `TimeMid` in seconds, and preferably `TimeStart,TimeEnd` in seconds. TAC columns in **Bq/mL**, decay-corrected to the beginning of their respective single-tracer scans.
- A CSV mapping `label_id,region_name,fdg_column,pbr_column` to the exact actual spreadsheet column names. IDs must match the NIfTI labels. See `docs/example_brain_roi_mapping.csv` (example IDs are illustrative only). Missing labels stay zero.
- The PBR28 table's `TimeStart,TimeEnd` determines output frames unless you supply `--frame-edges-csv` with contiguous frame boundaries in seconds.

To generate an **editable mapping template** with matching header names:

```bash
python -m sim.make_brain_roi_mapping \
  --atlas "/private/brain_labels.nii.gz" \
  --fdg-rois "/private/FDG_brain_TACs.xlsx" \
  --pbr-rois "/private/PBR_brain_TACs.xlsx" \
  --atlas-lut "/private/atlas_label_names.csv" \
  --out results/brain_roi_mapping_template.csv
```

`--atlas-lut` is optional and must contain `label_id,region_name`. Only exact normalized-name matches are filled automatically. **Inspect and edit mapping blanks** before running.

Make a 60-minute synthetic mixture with FDG injected 5 minutes after PBR28:

```bash
python -m sim.populate_brain_phantom \
  --atlas "/private/brain_labels.nii.gz" \
  --fdg-rois "/private/FDG_brain_TACs.xlsx" \
  --pbr-rois "/private/PBR_brain_TACs.xlsx" \
  --mapping-csv "results/brain_roi_mapping_template.csv" \
  --output-dir results/brain_phantom_v1 \
  --duration-min 60 \
  --fdg-offset-min 5 \
  --preview-min 5 20 50 \
  --save-4d
```

Output: `phantom_roi_tacs.csv`, `phantom_preview.png`, `phantom_roi_curves.png`, `phantom_metadata.json`. With `--save-4d`: `fdg_4d.nii.gz`, `pbr_4d.nii.gz`, `mixture_4d.nii.gz`. Images share the atlas geometry; for actual-PET comparison, register/resample atlas labels to the PET grid first using nearest-neighbor methods.

**Optional noisy mixture:**

```bash
python -m sim.populate_brain_phantom \
  --atlas "/private/brain_labels.nii.gz" \
  --fdg-rois "/private/FDG_brain_TACs.xlsx" \
  --pbr-rois "/private/PBR_brain_TACs.xlsx" \
  --mapping-csv "results/brain_roi_mapping_template.csv" \
  --noise-calibration "results/FDG_brain_noise/noise_region_summary.csv" \
  --output-dir results/brain_phantom_noisy \
  --duration-min 60 --fdg-offset-min 5 --save-4d
```

The alternative `--noise-sensitivity 0.003` selects an **uncalibrated** surrogate effective-Poisson coefficient; do not use it as a Panorama GS measurement. Calibrated labels must correspond between atlas and noise CSV. The voxelwise random draws are independent: no realistic correlated reconstruction noise, PSF, partial volume, attenuation, scatter or random corrections are modeled. We recommend frame-dependent validation before treating the simulated noisy images as realistic.

The scripts **reverse decay correction independently** for FDG and PBR28 using their respective physical half-lives, shift FDG by the specified offset, integrate frame means, and sum the resulting physical activities. The source scan beginning is assumed to coincide with tracer injection; verify this for your datasets. Temporal interpolation of discrete ROI TACs is approximate and cannot recover missing first-pass detail. Neither synthetic components nor mixtures constitute genuine dual-tracer measured images or demonstrated TCN separation performance.

**Data handling:** keep source PET, registered atlas, human-derived TAC tables and phantom output under private directories or ignored `results/` paths. Public GitHub contains code and synthetic-only tests.
