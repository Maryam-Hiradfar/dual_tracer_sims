# ROI-level dual-tracer pilot (no phantom and no full DICOM conversion)

The fastest presentation-ready experiment for our FDG/PBR28 project:

1. Load matched brain ROI TAC columns from **separately acquired** FDG and PBR28 CSV/XLSX.
2. Rebin/interpolate to the project's preferred dynamic frame schedule (2s×20,
   5s×12, 10s×12, 20s×12, 30s×8, 60s×6, 120s×5,
   300s×8, 600s×4), **truncated to 60 minutes**.
3. Shift FDG injection by 0, 5, or 10 min relative to PBR28 at t=0.
4. Apply **physical radioactive decay** from each tracer's own reference
   scan time, average in each synthetic output frame, and sum.
5. Test illustrative ROI measurement noise; export TACs, plots and a
   one-slide PowerPoint.

**Neither the synthetic mixture nor the template-fit test is proof of blinded
dual-tracer separation.** True TCN evaluation needs a model checkpoint and
a held-out test protocol.

## Quick start: from measured ROI TACs

The two TAC files must have `TimeMid` (seconds) and ROI-named columns (Bq/mL).
Preferably include `TimeStart,TimeEnd` in seconds, to establish data coverage.
Signals must be **decay-corrected to each source scan start**; the workflow
assumes scan start and injection coincide.

If column names are identical in FDG/PBR28, use:

```bash
python -m pip install numpy pandas scipy matplotlib openpyxl python-pptx pytest

python -m sim.roi_pilot \
  --fdg-rois "/private/FDG_brain_TACs.xlsx" \
  --pbr-rois "/private/PBR_brain_TACs.xlsx" \
  --rois Cerebellum Thalamus Insula \
  --output-dir results/roi_pilot_v1 \
  --duration-min 60 \
  --offsets-min 0 5 10 \
  --noise-cv60 0.05 0.10 \
  --noise-reps 50

python -m sim.roi_pilot_slide \
  --fig-dir results/roi_pilot_v1 \
  --out results/roi_pilot_v1/preliminary_ROI_slide.pptx
```

**Only use --rois names that appear in BOTH tables.** Omit the flag to
automatically use all exactly-matching column names and plot the first three.
To align differing names, pass a CSV with
`region_name,fdg_column,pbr_column` using `--roi-mapping`.

Outputs: `01_roi_pilot_figure.png` (ROI TACs, synthetic mixture and injection
timing comparison), `02_noise_sensitivity.png` (noisy mixture deviation
from clean mixture, **not separation error**), `roi_mixture_tacs.csv`,
`roi_noise_trials.csv`, `roi_noise_summary.csv`,
`oracle_template_fit_trials.csv`, and `pilot_metadata.json`.

The template fit uses both known reference curves as *input templates* to a
nonnegative least-squares regression; this optimistic test is **not blind
signal separation** and must not be compared directly with TCN validation.

The default "CV60" noise level is an **illustrative ROI-average-equivalent
CV at 10,000 Bq/mL and a 60-second frame**. Values 0.05 and 0.10 correspond
to nominal 5% and 10% CV at that reference point; true ROI noise must be
calibrated separately. The noise varies inversely with square root of
frame duration and source activity in this simple effective-count surrogate.
Do not equate ROI-average noise with independent **voxelwise** noise.

## Optional: single-ROI DICOM noise proxy, without giant 4D NIfTI

This samples only a few DICOM volumes around target frame durations.
Choose **one** reasonably homogeneous 3D white-matter ROI mask.
All DICOMs should be in one directory. The mask must already be in
the PET physical coordinate system.

```bash
python -m sim.sample_roi_dicom_noise \
  --pet-dicom-dir "/private/FDG_DICOM" \
  --roi-mask "/private/FDG_NeuroSegmenter_split/WhiteMatter.nii.gz" \
  --timing-csv "/private/FDG_frame_timings.csv" \
  --output-dir results/FDG_single_ROI_noise \
  --target-durations-sec 5 30 60 300
```

This reads DICOM headers, selects **one 3D time volume** at each requested
duration, and evaluates voxelwise spatial residual variation inside the
selected ROI. It does not convert all frames into a 4D NIfTI.

The output coefficient `median_equivalent_sensitivity_cps_per_bq` is for
**individual reconstructed voxels**, not a ready-made ROI-average noise
coefficient. A true ROI-level calibration requires understanding correlated
noise and the effective independent voxel count. White-matter heterogeneity,
partial-volume effects, smoothing and motion can bias spatial-residual CV.
Use this as an exploratory physics check only, not as an exact scanner
sensitivity or a direct reference for the ROI pilot noise setting.

For scientifically defensible uncertainty, independently reconstructed
split-half list-mode data are preferable.

## Future spatial/voxelwise extension

`sim.populate_brain_phantom` and `sim.estimate_pet_noise` remain available.
Once the ROI pilot and TCN evaluation are stable, use real registered
NeuroSegmenter labels and calibrated noise for a synthetic brain image
experiment. Do not commit human-derived TAC, DICOM, or generated data to
the public repository.
