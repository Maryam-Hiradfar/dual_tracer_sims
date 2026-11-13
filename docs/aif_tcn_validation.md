## Complete test-set PDF (added)

The usual `python -m ml.evaluate_tcn_prototype` command now writes
`evaluation/all_test_cases.pdf` alongside the aggregate diagnostic
PNG plots. Every held-out test sample is included exactly once,
grouped by FDG injection offset (0/5/10 min by default), four cases
per page. Each case plots noisy and clean mixture, FDG/PBR28 true
and predicted curves, and per-tracer RMSE and normalized percent
RMSE. Legends are outside the data axes.

You can regenerate the PDF **without any TCN retraining or inference**
from the already saved `test_predictions.npz`:

```bash
python -m ml.tcn_all_cases_pdf \
  --data-dir "/path/to/TCN_parametric" \
  --evaluation-dir "/path/to/TCN_parametric_results/evaluation" \
  --cases-per-page 4
```

With 1,000 generated samples and an 80/10/10 split, this produces
roughly 100 test plots across approximately 26–28 PDF pages,
depending on offset-group sizes. Note that it covers **test.npz** only,
not the training/validation examples.

# Expanded dual-tracer TCN validation: percentage error and AIF separation

The same causal `ml.tcn.Model` supports two **separate** parametric benchmarks:

- `--target-mode tissue` (default): analytic AIFs -> 2TCM -> physical
  tissue activity mixture -> predicted FDG/PBR28 tissue TACs.
- `--target-mode aif`: varied analytic AIFs -> physically decaying blood
  activity mixture -> predicted FDG/PBR28 AIF TACs. **No 2TCM
  transformation in the target**. The dataset still carries synthetic
  2TCM curves for comparison, but TCN targets are AIFs.

Both use exact PBR28 measurement frame boundaries read from the
`--frame-csv` file (TimeStart and TimeEnd in seconds). Both use
model-generated activity; no measured tissue TAC interpolation.

## Generate and train AIF benchmark

Assume `configs/IBDH_004_fit_params.json` was generated using
`python -m sim.parametric_tcn_dataset fit-aif` (see
`docs/parametric_tcn_benchmark.md`).

```bash
python -m sim.parametric_tcn_dataset generate \
  --params configs/IBDH_004_fit_params.json \
  --frame-csv "/Volumes/Neda 1TB/IBD_R01_prelim_analysis_results/IBDH_004_PBR/Aorta PBR.csv" \
  --target-mode aif \
  --output-dir "/Volumes/Neda 1TB/20260615/IBDH_004_FDG_IBDH_004_FDG_155716/TCN_parametric_AIF" \
  --duration-min 60 \
  --fdg-offsets-min 0 5 10 \
  --noise-cv60 0.05 \
  --n-samples 1000

python -m ml.train_tcn_prototype \
  --data-dir "/Volumes/Neda 1TB/20260615/IBDH_004_FDG_IBDH_004_FDG_155716/TCN_parametric_AIF" \
  --output-dir "/Volumes/Neda 1TB/20260615/IBDH_004_FDG_IBDH_004_FDG_155716/TCN_parametric_AIF_results" \
  --epochs 30

python -m ml.evaluate_tcn_prototype \
  --data-dir "/Volumes/Neda 1TB/20260615/IBDH_004_FDG_IBDH_004_FDG_155716/TCN_parametric_AIF" \
  --checkpoint "/Volumes/Neda 1TB/20260615/IBDH_004_FDG_IBDH_004_FDG_155716/TCN_parametric_AIF_results/best_tcn_prototype.pt" \
  --output-dir "/Volumes/Neda 1TB/20260615/IBDH_004_FDG_IBDH_004_FDG_155716/TCN_parametric_AIF_results/evaluation"
```

## Percent error metrics

`test_report.json -> physical_metrics.by_tracer.FDG/PBR28` contains:

- `percent_rmse` = 100 x RMSE / RMS(true tracer TAC)
- `percent_mae` = 100 x MAE / RMS(true tracer TAC)
- `reference_rms` = RMS(true tracer TAC) in Bq/mL

We deliberately use **global tracer RMS concentration** rather than
ordinary pointwise MAPE, which explodes near zero activity in the
pre-injection frames. `percent_error_over_time.png` shows framewise
RMSE and MAE divided by the tracer's global RMS concentration.
`percent_error_by_offset.csv` reports offset-specific percent RMSE,
normalized to that offset group's RMS ground truth. Legends are outside
axes.

The tissue TCN checkpoint **does not** automatically become an AIF
separator; train a separate AIF-mode checkpoint.

## Second-subject out-of-subject synthetic challenge

Fit the analytic AIF model to a *different* subject's FDG and PBR28
arterial datasets with `fit-aif`, producing a second JSON.
Then generate a second test dataset from **only subject B parameters**
using `generate`, with the **same target-mode** as the model trained
on subject A. Run `ml.evaluate_tcn_prototype` with **subject A's
checkpoint** and **subject B's data-dir**. No retraining/fine-tuning.
Keep the framing scheme comparable. The evaluation's test split is
held out from subject B's generated dataset, but is **entirely
unseen relative to subject A training** provided subject B was not
used in training or parameter selection.

This tests transport from a parameter model calibrated on subject A to
a different subject's calibrated forward model—not separation of an
actually simultaneously acquired dual-tracer scan, which is unavailable.

## ROI-wise images

**Yes**, but distinguish image outputs:
1. ROI-grid TAC plots: one panel per anatomical ROI, with FDG/PBR28
   predicted and ground truth curves, using region-specific kinetic
   fits and forward simulations.
2. ROI-parametric maps: assign predicted ROI values (e.g., FDG/PBR28
   concentration, area under curve, fit error) to each spatially
   registered ROI segmentation and save aligned NIfTI maps.
3. Full voxelwise separation: requires 4D reconstructed PET plus a
   voxelwise trained model or spatially informed inference, and is
   NOT achieved by copying ROI predictions into voxels.

A TCN trained on synthetic mixtures from a single parameterized region
does not establish reliable cross-ROI image inference. First extend
region-specific parameter distributions and validation.

## Tests

```bash
python -m pytest tests/test_parametric_tcn_dataset.py tests/test_tcn_diagnostic_plots.py -v
```
