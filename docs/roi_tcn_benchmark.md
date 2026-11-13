## Full TCN evaluation figures (added)

After pulling the latest branch, **re-run only**
`python -m ml.evaluate_tcn_prototype` against the saved TCN checkpoint.
Retraining is not needed solely to regenerate figures.

Evaluation now saves:
`test_prediction_examples.png`,
`component_predictions.png`,
`component_residuals.png`,
`prediction_parity.png`,
`error_over_time.png`,
`error_histograms.png`,
`error_by_offset.png`,
`mixture_consistency.png`, and, when
`history.json` is next to the checkpoint,
`training_curves.png`. Legends are outside plotting axes.

**Aorta CSVs contain BLOOD input functions, not brain tissue TACs.**
To obtain TCN predictions of tissue kinetics, re-generate the
dataset using paired ROI tissue TAC tables (same region names or
`--roi-mapping`). No code can turn the aorta input function
into measured tissue activity by relabeling figures.

The branch's evaluation script previously generated only a single
three-panel overlay figure. These additions expand that script,
but they are not claimed to reproduce unavailable student-authored
figures that may live outside the tracked repository.

## Source timing and activity scale defaults (updated)

As of this revision, `sim.generate_roi_tcn_dataset` defaults to
`--frame-schedule measured --scale-range 1 1`.

- **Timing:** read `TimeStart` and `TimeEnd` in seconds from the
  **PBR28 CSV** and use those native reconstructed time frames as
  the common output timeline. It rejects gaps/overlaps and requires
  coverage for the requested `--duration-min`. FDG uses its own
  original `TimeMid` and is rebinned/interpolated onto the PBR28
  frames after the chosen FDG injection shift.
- **Scales:** preserve the two measured concentrations in Bq/mL;
  no random amplitude/dose scaling by default. Tracers are
  shifted and physically decay-corrected back to their
  physical-activity mixture domain; this naturally changes the
  numerical TAC magnitude.
- **Optional augmentation:** `--scale-range 0.85 1.15` independently
  scales FDG/PBR28 concentrations for additional synthetic variation.
  This is optional, **not** measured concentration.
- **Alternative framing:** `--frame-schedule preferred` selects the
  earlier proposed 2s/5s/10s/... synthetic schedule instead.

Current example using the user's aortic blood TACs (replace paths):

```bash
python -m sim.generate_roi_tcn_dataset \
  --fdg-rois "/private/FDG Aorta.csv" \
  --pbr-rois "/private/Aorta PBR.csv" \
  --output-dir datasets/aorta_measured_framing \
  --duration-min 60 \
  --offsets-min 0 5 10 \
  --frame-schedule measured \
  --scale-range 1 1 \
  --noise-cv60 0.05 \
  --n-samples 1000
```

**Important:** with one measured aortic TAC per tracer, fixed scales
do not create new kinetic shapes: stochastic noise and injection delays
create the only sample variation. This benchmark is a technical
feasibility test, not independent biological generalization.
The 5% noise is still illustrative, not measured.

# ROI TAC -> EXISTING causal TCN benchmark (revised timing and empirical noise)

This is **the student's existing** `ml.train_tcn_prototype` /
`ml.evaluate_tcn_prototype` benchmark. No phantom, NNLS,
or new inference model. The only new dataset-preparation entry point is
`sim.generate_roi_tcn_dataset`. Source data remain local.

Input FDG and PBR28 CSV/XLSX: TimeMid (seconds), preferably TimeStart
and TimeEnd, plus brain ROI columns with activities in Bq/mL,
decay-corrected to each tracer scan start. ROI names must match
exactly, or specify an explicit mapping CSV
(`region_name,fdg_column,pbr_column`). Do not commit real data.

### Step 1: preliminary empirical noise from ONE representative ROI

The optional `sim.sample_roi_dicom_noise` script reads only a few chosen
FDG PET DICOM volumes and one already registered ROI mask:

```bash
python -m sim.sample_roi_dicom_noise \
  --pet-dicom-dir "/private/FDG_PET_DICOM" \
  --roi-mask "/private/FDG_WhiteMatter.nii.gz" \
  --timing-csv "/private/FDG_frame_timings.csv" \
  --output-dir results/reference_noise \
  --target-durations-sec 5 30 60 300
```

Result: `results/reference_noise/single_roi_noise_samples.csv`. The
sampled spatial-residual CV estimates **voxel** noise, not direct ROI
mean noise. To calibrate ROI measurement noise for the simulation,
supply an estimated **effective number of independent voxels**.
This is a modeling assumption; true voxel correlations and spatial
heterogeneity could make it substantially different from the raw
ROI voxel count.

### Step 2: generate TCN-compatible data from empirical brain ROI TACs

```bash
python -m sim.generate_roi_tcn_dataset \
  --fdg-rois "/private/FDG_brain_TACs.xlsx" \
  --pbr-rois "/private/PBR28_brain_TACs.xlsx" \
  --output-dir datasets/roi_tcn_v1 \
  --duration-min 60 \
  --offsets-min 0 5 10 \
  --noise-samples-csv results/reference_noise/single_roi_noise_samples.csv \
  --effective-roi-voxels 10 \
  --n-samples 1000
```

The `10` above is only an **illustrative** effective voxel count;
use a defensible value and vary it to show sensitivity.
For a preliminary smoke test without image-derived calibration, omit
`--noise-samples-csv` and `--effective-roi-voxels` and instead use
`--noise-cv60 0.05`: this is illustrative, not calibrated.

The preferred frame schedule is:
2s×20, 5s×12, 10s×12, 20s×12, 30s×8,
1min×6, 2min×5, 5min×8, 10min×4,
truncated at 60 minutes. Both source TACs must cover the requested
simulation interval (allowing for injection offsets).

Physical decay for FDG and PBR28 is applied independently to each
source TAC. Synthetic outputs use the same NPZ keys as the previous
TCN benchmark, in the same channel order [FDG,PBR28]. Dose-like
relative scaling from .85 to 1.15 supplies sample diversity.

### Step 3: train and evaluate the existing student TCN

```bash
python -m ml.train_tcn_prototype \
  --data-dir datasets/roi_tcn_v1 \
  --output-dir ml/experiments/roi_tcn_v1 \
  --epochs 30

python -m ml.evaluate_tcn_prototype \
  --data-dir datasets/roi_tcn_v1 \
  --checkpoint ml/experiments/roi_tcn_v1/best_tcn_prototype.pt \
  --output-dir ml/experiments/roi_tcn_v1/evaluation
```

Outputs: `test_prediction_examples.png`, `test_report.json`,
and `test_predictions.npz`. The evaluator plots **TCN-predicted FDG
and PBR28** versus synthetic ground truth for representative
0/5/10-minute offsets. Display the test prediction figure and FDG/PBR28
test errors on a preliminary slide.

### Essential validation caveats

- The noise coefficient comes from one selected FDG ROI; transfer to
  PBR28 and other ROIs is a simplifying assumption, not validated.
- ROI TAC averages suppress spatial noise; converting a voxel CV
  into ROI-mean CV requires an effective independent voxel count.
- Random train/test variants can share the SAME source ROI TACs,
  so performance does NOT demonstrate unseen-subject or unseen-ROI
  generalization. Split by subject/ROI later for a rigorous validation.
- Rebinning frame-average TAC measurements by midpoint interpolation
  is approximate, especially around early bolus dynamics.
- Physical decay assumes single-tracer TACs are decay-corrected to
  injection/scan-start time.
- Existing TCN remains unchanged and evaluates a **synthetic**
  physically-decaying mixed signal, not a true simultaneously
  acquired dual-tracer scan.

## Tests

```bash
python -m pytest tests/test_generate_roi_tcn_dataset.py -v
```
