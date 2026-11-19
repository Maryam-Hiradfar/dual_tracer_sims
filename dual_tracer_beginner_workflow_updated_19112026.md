# Dual-Tracer PET Simulation and TCN Benchmark
## Beginner workflow: from a fresh checkout to evaluated FDG/PBR28 separation

**Repository:** `Maryam-Hiradfar/dual_tracer_sims`  
**Working branch:** `feature/bqml-dose-calibration`  
**Guide date:** 18 November 2026  
**Intended reader:** New lab member or collaborator who has not worked with this codebase.

> **Recommended workflow:** Fit smooth AIF model parameters from paired single-tracer aortic data (once), generate synthetic FDG/PBR28 tissue activity on **actual PET frame boundaries**, train the existing causal TCN, and evaluate the held-out synthetic samples. The alternatives below have different assumptions and must not be mixed accidentally.

## 1. What this project does

The TCN learns a supervised mapping from a **single noisy mixed time–activity curve (TAC)** to two separate, modeled physical-activity curves:

`modeled FDG + modeled PBR28 -> physical radioactive decay -> frame integration -> noisy mixture -> TCN -> estimated FDG and PBR28`

The existing model is in `ml/tcn.py`; training and evaluation live in `ml/train_tcn_prototype.py` and `ml/evaluate_tcn_prototype.py`. Model input has shape `(number_of_samples, 1, number_of_frames)`; targets have shape `(number_of_samples, 2, number_of_frames)` in channel order **FDG, PBR28**. The TCN uses causal, dilated 1D convolutions; the standard training example uses an 80/10/10 train/validation/test split (800/100/100 for 1,000 simulations).

**These are synthetic separation benchmarks, not actual acquired simultaneous dual-tracer data.** A random split of simulations derived from one subject's parameter estimates is not independent-subject validation.

## 2. Choose the dataset generator

| Workflow | Command entry point | Inputs | Target | Main limitation |
|---|---|---|---|---|
| **A — Parametric tissue (recommended)** | `sim.parametric_tcn_dataset` | Fitted analytic AIF parameters, kinetic model, real frame times | Synthetic FDG/PBR28 tissue TACs | PBR28 kinetics currently illustrative; noise uncalibrated |
| **B — Parametric blood/AIF** | Same generator with `--target-mode aif` | Fitted analytic AIF parameters, real frame times | Synthetic FDG/PBR28 AIFs | Blood separation is **not** tissue separation |
| C — Empirical ROI (alternative) | `sim.generate_roi_tcn_dataset` | Measured FDG/PBR28 tissue-ROI CSVs | Reframed empirical tissue TAC mixtures | Interpolates measured frame means; avoid if no interpolation is desired |
| D — Older empirical Bq/mL model | `sim.generate_bqml_dataset` | Measured aortic IDIF CSVs | Blood or approximate 2TCM tissue | Interpolates empirical IDIFs; approximate kinetics |

Use **A** for the current model-based tissue benchmark. For the distinct blood-input experiment use **B** and **train a separate checkpoint**. Workflows C and D are retained for comparison; they are not prerequisites for A/B.

## 3. Setup on Mac / Linux

Open a terminal in the clone directory. Keep all participant data and generated fitted parameters on an approved private drive or approved institutional system.

```bash
# Clone once, or skip this if your lab already has a clone.
git clone https://github.com/Maryam-Hiradfar/dual_tracer_sims.git
cd dual_tracer_sims

git fetch origin
git switch feature/bqml-dose-calibration
git pull --ff-only origin feature/bqml-dose-calibration

# Activate an existing project environment, or create one:
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas matplotlib torch pytest

# Make sure the modules are importable from the repo root.
python -c 'import numpy, scipy, pandas, matplotlib, torch; print("Environment OK")'
python -m sim.parametric_tcn_dataset --help
```

The command above installs the **core dependencies used in this guide**, not every optional imaging dependency. DICOM and NIfTI paths additionally need `pydicom`, `nibabel`, etc. If your lab provides an established environment, use it instead of making another one. Commands assume execution **from the repo root** (`python -m ...`). Use quoted paths when a drive name contains spaces.

## 4. Data inputs and formats

**For fitting AIF parameters (once):** provide separate FDG and PBR28 aorta CSVs. Both must have:

| Column | Meaning |
|---|---|
| `TimeStart` | Native PET frame start, **seconds** |
| `TimeEnd` | Native PET frame end, **seconds** |
| `aorta` | Frame-averaged aortic activity concentration, **Bq/mL** |

Example format (illustrative values only; do not replace actual measurements with these):

```csv
TimeStart,TimeEnd,aorta
0,5,1200
5,10,5500
10,20,10800
```

The AIF fit compares **integrated analytical model predictions** to measured frame averages; it does **not** interpolate the source TACs into the training sample timelines. The model is a fitted gamma-like bolus plus fast and slow exponential terms, **not** the older Feng model. All analytic-curve amplitudes are in Bq/mL. Both files should have known, consistent decay-correction references; the model assumes scan-start/injection-time alignment for the reference data.

**For generating simulated data:** `--frame-csv` only needs `TimeStart` and `TimeEnd`, in seconds. Frame boundaries must start at zero, be contiguous and increasing, and cover the duration requested. Use the original PBR28 frame schedule as the common output timeline for both tracers. The simulator creates a smooth, continuous numerical model and integrates it over **the exact input frame boundaries**; it does not interpolate measured ROI activity values.

## 5. Quick start: parametric tissue simulation (A)

### 5.1 Fit the analytical AIF model to aorta measurements

Change the example file paths to **your approved local input files**. The output JSON is an editable parameter file.

```bash
python -m sim.parametric_tcn_dataset fit-aif \
  --fdg-aif-csv "/path/to/FDG Aorta.csv" \
  --pbr-aif-csv "/path/to/Aorta PBR.csv" \
  --out-params "/path/to/private/params_subject_A.json"
```

**Check:** the command prints FDG and PBR28 AIF fit RMSE (Bq/mL). Inspect the resulting JSON, check plausible parameter values, and review fitting residuals before using it for scientifically quantitative conclusions. A low optimizer error does not guarantee physiological identifiability. Do not commit subject-specific fitted parameters to a public GitHub repository.

### 5.2 Review and adjust the parameter JSON

Important fields (the generated JSON includes these):

| JSON field | Default / role | Status |
|---|---|---|
| `aif.FDG`, `aif.PBR28` | Bolus amplitude, peak, shape, delay, fast/slow tails | **Fitted from measured aorta frame averages** |
| `kinetics.FDG` | K1≈0.14697, k2≈0.11958, k3≈0.04211, k4=0 | Prior **single-region** FDG fit; not population distribution |
| `kinetics.PBR28` | K1=0.18, k2=0.25, k3=0.04, k4=0.01 | **Illustrative**, not a measured subject fit |
| `vascular_fraction.FDG` | ~0.08085 | Prior FDG fit |
| `vascular_fraction.PBR28` | 0.05 | Illustrative |
| `reference_doses_mci` | FDG 4.91; PBR28 4.98 | Reference injected activities |
| `dose_ranges_mci` | Defaults to fixed reference doses | Editable dose augmentation |
| `kinetic_variation_fraction` | 0.10 (±10%) | **Assumed** perturbation, not measured biological variability |
| `aif_variation_fraction` | 0.05 (±5% amplitude) | **Assumed** perturbation |
| `fdg_extra_delay_sec` | ~5.78 s | Fitter sign/convention needs verification |

For an initial comparison use the JSON unchanged, but label the PBR28 kinetics and noise settings **simulation assumptions**. PBR28 uses **whole-blood aorta** as the input surrogate, not metabolite-corrected parent-plasma activity.

### 5.2a Where do the AIF and tissue kinetic parameters come from?

**Current simulation flow:** measured FDG/PBR28 aortic frame averages → fitted analytical AIF parameters → forward two-tissue compartment model (2TCM) → synthetic tissue TACs → physical decay and exact measured frame integration → noisy mixture → TCN. The resulting tissue TACs are *not* interpolated measured ROI TACs.

- **FDG:** reference tissue parameters are based on a prior real **single-brain-ROI fit**: K1≈0.14697, k2≈0.11958, k3≈0.04211, k4=0, vascular fraction vb≈0.08085, and additional delay≈5.78 seconds. They are not an empirically estimated regional or population parameter distribution.
- **PBR28:** reference K1=0.18, k2=0.25, k3=0.04, k4=0.01, vb=0.05 are **illustrative**, not fitted to PBR28 tissue data. The measured PBR28 aorta is whole blood, not metabolite-corrected parent plasma.
- **Variation:** kinetic rate constants are varied independently by ±10%, and AIF amplitudes by ±5%, around reference values. These ranges are assumed for synthetic training, not derived from subject-to-subject or ROI-wise measurements. FDG 4.91 mCi and PBR28 4.98 mCi are the reference injection doses.
- **Frame schedule:** uses the real measured PBR28 `TimeStart` / `TimeEnd` boundaries; physical decay and a currently assumed noise model are applied.

**What remains missing:** tissue kinetic parameter fits for **both tracers, across multiple anatomical ROIs**, together with an empirically informed distribution for generating realistic region- and subject-level variability.

### 5.3 Generate 1,000 synthetic tissue mixtures

```bash
python -m sim.parametric_tcn_dataset generate \
  --params "/path/to/private/params_subject_A.json" \
  --frame-csv "/path/to/Aorta PBR.csv" \
  --target-mode tissue \
  --output-dir "/path/to/private/results/subject_A_tissue_dataset" \
  --duration-min 60 \
  --fdg-offsets-min 0 5 10 \
  --noise-cv60 0.05 \
  --n-samples 1000 \
  --seed 48
```

This integrates the 2TCM on an approximately one-second numerical grid, applies physical decay using FDG (109.8 min) and PBR28 (20.4 min) half-lives, frame-averages using measured frame boundaries, sums the components, and simulates count-equivalent Poisson noise. **`--noise-cv60 0.05` is not scanner-calibrated:** it means an assumed 5% ROI-average CV at 10,000 Bq/mL and 60 seconds. Default output is 800 training, 100 validation, 100 test examples.

### 5.4 Verify generated files before training

Expected files inside `--output-dir`:

```text
train.npz       val.npz       test.npz
dataset_metadata.json
train_metadata.json  val_metadata.json  test_metadata.json
```

Run:

```bash
python - <<'PY'
import json, numpy as np
from pathlib import Path
root = Path('/path/to/private/results/subject_A_tissue_dataset')
meta = json.loads((root / 'dataset_metadata.json').read_text())
print('Target mode:', meta.get('target_mode'))
print('Number of frames:', len(meta['frame_midpoints_min']))
print('First five frame durations (s):',
      [round(x*60,3) for x in meta['frame_durations_min'][:5]])
for split in ('train','val','test'):
    with np.load(root / f'{split}.npz') as z:
        print(split, z['X_measured_tavg_n1_d1'].shape,
              z['Y_phys_tavg_n0_d1'].shape)
        assert z['Y_phys_tavg_n0_d1'].shape[1] == 2
PY
```

Expected output shapes for **N** samples and **T** native frames: input `(N,1,T)`, targets `(N,2,T)`; output channels are **FDG first, PBR28 second**. Keep this same T and timing when evaluating a separately trained checkpoint.

## 6. Train the TCN and evaluate predictions

```bash
python -m ml.train_tcn_prototype \
  --data-dir "/path/to/private/results/subject_A_tissue_dataset" \
  --output-dir "/path/to/private/results/subject_A_tissue_model" \
  --epochs 30

python -m ml.evaluate_tcn_prototype \
  --data-dir "/path/to/private/results/subject_A_tissue_dataset" \
  --checkpoint "/path/to/private/results/subject_A_tissue_model/best_tcn_prototype.pt" \
  --output-dir "/path/to/private/results/subject_A_tissue_model/evaluation"
```

Training saves `best_tcn_prototype.pt` (chosen by validation loss) and `history.json`. Evaluation saves `test_report.json`, `test_predictions.npz`, and diagnostic images including component predictions, residuals, parity, mixture consistency, error by time/offset, **percent error**, and training curves when history exists. It also writes **`all_test_cases.pdf`**: every held-out test sample grouped by FDG injection offset, with per-tracer absolute and percentage RMSE. The representative TAC panel shows **one test case per 0/5/10-minute offset**. Aggregate metrics use all held-out test cases.

For percentage error, the evaluator uses `100 × RMSE / RMS(true tracer)` (and an analogous normalized MAE), **not** framewise MAPE, to avoid instability around zero pre-injection activity. Metrics appear at `physical_metrics.by_tracer.FDG/PBR28` in `test_report.json`.

If you only need to regenerate the all-cases PDF using saved evaluation arrays (no retraining):

```bash
python -m ml.tcn_all_cases_pdf \
  --data-dir "/path/to/private/results/subject_A_tissue_dataset" \
  --evaluation-dir "/path/to/private/results/subject_A_tissue_model/evaluation" \
  --cases-per-page 4
```

## 7. Alternative experiment: train an AIF separator (B)

AIF mode is intentionally **different from tissue mode**. It uses the varied analytical AIFs themselves as FDG/PBR28 targets, rather than 2TCM tissue activity. It still uses measured frame timing and physical decay. Keep datasets, checkpoints and evaluation directories separate.

```bash
python -m sim.parametric_tcn_dataset generate \
  --params "/path/to/private/params_subject_A.json" \
  --frame-csv "/path/to/Aorta PBR.csv" \
  --target-mode aif \
  --output-dir "/path/to/private/results/subject_A_aif_dataset" \
  --duration-min 60 --fdg-offsets-min 0 5 10 \
  --noise-cv60 0.05 --n-samples 1000

python -m ml.train_tcn_prototype \
  --data-dir "/path/to/private/results/subject_A_aif_dataset" \
  --output-dir "/path/to/private/results/subject_A_aif_model" --epochs 30

python -m ml.evaluate_tcn_prototype \
  --data-dir "/path/to/private/results/subject_A_aif_dataset" \
  --checkpoint "/path/to/private/results/subject_A_aif_model/best_tcn_prototype.pt" \
  --output-dir "/path/to/private/results/subject_A_aif_model/evaluation"
```

**Never evaluate an AIF dataset with a tissue-trained TCN and describe the result as a matched task.** They predict different biological targets.

## 8. Optional, older alternatives and imaging preparation

**Empirical tissue ROI workflow:** `sim.generate_roi_tcn_dataset` accepts paired FDG/PBR28 CSV/XLSX ROI activity columns, matching ROI names or `--roi-mapping`. It uses measured PBR28 framing by default and measured Bq/mL scale (`--scale-range 1 1`), **but it interpolates input frame-average TAC values** when resampling. See `docs/roi_tcn_benchmark.md`. Choose this only when empirical TAC interpolation is acceptable; it is **not** the current preferred parametric workflow.

**Older dose-referenced IDIF workflow:** `sim.generate_bqml_dataset` uses measured IDIFs and optional approximate tissue kinetics; see `docs/bqml_calibration.md` and `sim.validate_bqml`. This is retained for comparison. It also interpolates empirical aortic measurements.

**Noise estimation and imaging:** `sim/sample_roi_dicom_noise.py` samples one brain ROI across selected enhanced-PET DICOM frames. `sim/estimate_pet_noise.py`, `sim/dicom_dynamic_pet.py` and `sim/enhanced_pet.py` support more extensive PET noise/image preparation. High-volume 4D conversions may consume tens of GB and require registration and frame-order QC. See `docs/brain_phantom_and_noise.md`. **Neither ROI atlas population nor voxelwise FDG/PBR28 separation is automatically implemented by training the ROI TCN.**

## 9. Tests and debugging

Run from the repository root:

```bash
python -m pytest \
  tests/test_parametric_tcn_dataset.py \
  tests/test_tcn_percent_metrics.py \
  tests/test_tcn_diagnostic_plots.py \
  tests/test_tcn_all_cases_pdf.py -v
```

| Symptom | Likely cause | What to do |
|---|---|---|
| `ModuleNotFoundError: sim` / `ml` | Running outside repo root | `cd dual_tracer_sims` and use `python -m ...` |
| `No such file or directory` | Incorrect private drive/path, unmounted external volume | Quote the complete path; confirm `ls "/path/to/file.csv"` |
| `TimeStart`/`TimeEnd` error | Missing columns, non-contiguous times or insufficient coverage | Inspect source CSV. Values are **seconds**, not minutes. |
| `No TAC activity column` | Aorta CSV used with empirical ROI loader instead of intended mode | For the recommended workflow, use `fit-aif`, not `generate_roi_tcn_dataset` |
| Shape mismatch between model and test | Different number of native frames | Use same frame schedule for a frozen TCN checkpoint or retrain for new T |
| Plots still look unchanged | Local branch not updated, or old output files opened | `git status -sb`; `git pull`; check output timestamps; rerun evaluator |
| Huge percentage error before injection | Dividing by near-zero values | Use normalized percentage RMSE, not pointwise MAPE |
| Missing trained checkpoint | Training not run or wrong `--output-dir` | Find `best_tcn_prototype.pt`; use exact path |
| Import error for `pydicom`/`nibabel` | Optional imaging modules not installed | Install appropriate optional imaging dependencies |

**Important current presentation-style caveat:** A `finalize_slide_figure` helper exists in `ml/tcn_plot_style.py`, but at the time this guide was written it was not called by the figure exporters. The plotted PNGs may therefore not fully reflect the intended extra boldness. Do not assume slide styling is guaranteed until visually checked.

## 10. Scientific / data-security checklist before reporting results

- Confirm `TimeStart` and `TimeEnd` are **seconds** and activity is **Bq/mL**.
- Verify whether input AIF data are decay corrected, and the injection/scan-start reference convention.
- Check AIF fit residuals, especially the first-pass bolus peak; fitted parameters can hit bounds.
- Document FDG 4.91 mCi and PBR28 4.98 mCi **reference** injections and any dose variation applied.
- Clearly label PBR28 whole-blood-input 2TCM and kinetic constants as **approximations** until independently fitted/validated.
- Treat 5% CV (`--noise-cv60 0.05`) as an **assumption**, not calibrated scanner or ROI noise.
- Report dataset sample count, train/val/test split, injection offsets, native frame timing, random seed, target mode and model checkpoint.
- Do not interpret the synthetic test split as held-out participant validation. A new subject must be withheld from model calibration and tuning to meaningfully test generalization.
- Keep patient-derived AIFs, fitted JSONs, and simulation outputs in approved private storage. **Audit any previously committed fitted JSON**, including `configs/IBDH_004_fit_params.json`, for institutional data-governance compliance.

## 10a. Planned next steps and task ownership

These are **future work**, not finished features.

### 1. ROI-wise kinetic parameter estimation — Maryam

Create a reproducible fitting framework to derive kinetic parameters from measured single-tracer brain ROI TACs, separately for FDG and PBR28. The goal is to export a reviewed, machine-readable region-by-tracer parameter table containing K1, k2, k3, k4, blood fraction and delay when fitted, with fit quality, uncertainty, input-function definitions, and units. Compare predicted and observed TACs and inspect fitting diagnostics. These parameters would replace illustrative model constants in the simulation framework and eventually allow measured region-wise parameter variation.

**Deliverable:** ROI-wise FDG/PBR28 parameter tables and fitting QC that can be passed into synthetic dataset generation.

### 2. Populate a regional brain phantom and map TACs — Kayoung

Start from an atlas or set of region masks, assign **arbitrary kinetic parameters to each region** for the first prototype, and generate a known FDG TAC and PBR28 TAC per region. Map each regional TAC back to its spatial mask to generate 4D FDG and PBR28 brain phantom images, then construct a mixed dual-tracer series with the acquisition frame schedule, physical decay and synthetic noise. Verify region labels, voxel geometry, time/frame alignment, Bq/mL scale and mixed-signal consistency; save NIfTI outputs, parameter tables and QC images. This is a pipeline smoke test, not an anatomically validated simulation.

**Next iteration:** once Maryam provides the fitted parameters, substitute the measured ROI-specific estimates for the arbitrary regional values, then vary them in biologically informed ways and revalidate the simulated images.

**Deliverable:** phantom population and TAC-mapping framework with reproducible 4D images and QC; compatible with Maryam's parameter tables.

**Integration milestone:** first validate region mapping with arbitrary parameters, then upgrade to real fitted ROI kinetics. Full voxelwise TCN separation remains a separate future validation task.

## 11. File navigation for newcomers

| File | Purpose |
|---|---|
| `sim/parametric_tcn_dataset.py` | Recommended analytic AIF fitting, parametric tissue/AIF dataset generation |
| `sim/bqml_calibration.py` | Dose/decay constants and older empirical AIF forward model |
| `sim/generate_roi_tcn_dataset.py` | Alternate empirical ROI-TAC mixture generator |
| `ml/tcn.py` | Causal dilated residual TCN architecture |
| `ml/train_tcn_prototype.py` | Train, save checkpoint and training history |
| `ml/evaluate_tcn_prototype.py` | Run test inference, errors and diagnostic plots |
| `ml/tcn_diagnostic_plots.py` | Predictions, residuals, parity, percent-error figures |
| `ml/tcn_all_cases_pdf.py` | PDF with every held-out test case |
| `ml/tcn_plot_style.py` | Centralized plotting style settings |
| `docs/parametric_tcn_benchmark.md` | Technical details of the parameter-based workflow |
| `docs/aif_tcn_validation.md` | AIF-mode and evaluation documentation |
| `docs/brain_phantom_and_noise.md` | Imaging and PET noise preparation prototypes |
| `tests/test_parametric_tcn_dataset.py` | Dataset and simulator unit tests |

**Versioning note:** This guide describes the feature branch as inspected on 18 November 2026. Check `git status -sb` and `git rev-parse --short HEAD` and record the actual commit used for any reproducible experiment. Tests included in the repository are not a substitute for validation on the real acquisition files.
