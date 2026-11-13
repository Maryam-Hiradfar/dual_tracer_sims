# Real-framing, parameter-driven PET simulation -> existing TCN

This is **not** the measured ROI TAC interpolation pipeline.
It generates analytical arterial input functions from fitted parameters,
feeds them into a continuous-time two-tissue compartment model,
simulates physiological radioactive decay, and **only then** averages
the resulting activity over the **exact TimeStart/TimeEnd boundaries**
from a measured PET acquisition. Both FDG and PBR28 tissue component
targets and their observed noisy mixture are synthesized.

It retains the existing TCN NPZ interface; neither
`ml.train_tcn_prototype` nor `ml.evaluate_tcn_prototype`
needs changing.

## A. Fit AIF parameters once, from measured aortic CSVs

```bash
python -m sim.parametric_tcn_dataset fit-aif \
  --fdg-aif-csv "/Volumes/Neda 1TB/IBD_R01_prelim_analysis_results/IBDH_004_FDG/FDG Aorta.csv" \
  --pbr-aif-csv "/Volumes/Neda 1TB/IBD_R01_prelim_analysis_results/IBDH_004_PBR/Aorta PBR.csv" \
  --out-params "configs/IBDH_004_fit_params.json"
```

Inputs must have `TimeStart`, `TimeEnd` (seconds) and `aorta`
(Bq/mL), one row per native measured frame. The fit uses the **integral
of the analytic function across each measured frame**; measured frame
averages are not treated as instantaneous samples and not interpolated
onto a synthetic time grid. This is a new smooth gamma-bolus + dual
exponential AIF model; it is NOT the original Feng parameterization.

The resulting JSON is **editable**. It contains:
- Per-tracer AIF bolus amplitude/peak/shape and fast/slow exponentials;
  these are **fit to the CSV observations**.
- Reference injected activities **4.91 mCi FDG, 4.98 mCi PBR28**.
- FDG **single-region previously fitted** K1=.1469698332,
  k2=.1195751389, k3=.0421083673, k4=0,
  vb=.0808523134 and extra delay=5.780918 seconds.
- PBR28 **illustrative, not fitted to subject** K1=.18,
  k2=.25, k3=.04, k4=.01, vb=.05.
- Variation knobs: `kinetic_variation_fraction=.1`,
  `aif_variation_fraction=.05`. Dose ranges default to exactly
  reference dose, without random dose variation.

**Important:** AIF input from aorta is whole blood; it is NOT
metabolite-corrected PBR28 parent plasma. Therefore the simulated
PBR28 tissue signal is a plausible but not quantitatively calibrated
2TCM surrogate. The FDG fitter's delay sign and exact blood-fraction
mixing convention must be checked against the original fit before
claiming region-specific quantitative consistency.

## B. Generate 1,000 independent synthetic draws using native PBR frames

```bash
python -m sim.parametric_tcn_dataset generate \
  --params "configs/IBDH_004_fit_params.json" \
  --frame-csv "/Volumes/Neda 1TB/IBD_R01_prelim_analysis_results/IBDH_004_PBR/Aorta PBR.csv" \
  --output-dir "/Volumes/Neda 1TB/20260615/IBDH_004_FDG_IBDH_004_FDG_155716/TCN_parametric" \
  --duration-min 60 \
  --fdg-offsets-min 0 5 10 \
  --noise-cv60 0.05 \
  --n-samples 1000
```

You can pass a separate scanner frame-timing CSV if preferred; only
`TimeStart` and `TimeEnd` are read. Frame windows must be contiguous,
begin at t=0, and cover the chosen duration. To obtain the full
reference acquisition, adjust `--duration-min` accordingly.

The simulator integrates on a ~1-second uniform grid (RK4 compartment
solver) and frame-averages onto these exact measured frames. **No
measured tissue TAC values are interpolated or used as synthetic
targets.** The input function is generated anew from fitted analytic
parameters for each sample. The analytic AIF was calibrated once from
real measured frame averages.

The effective-count noise model is retained from the existing ROI
TCN workflow; `--noise-cv60 0.05` is a placeholder ROI-mean CV at
10,000 Bq/mL and 60 seconds, NOT scanner-calibrated noise.

**Accuracy boundaries:** There are independent stochastic samples,
not independent human subjects. The kinetic parameter ranges around
one real FDG fit do not constitute a validated brain-population
distribution. For PBR28 or regional physics, replace the JSON
parameters with independently fitted measurements.

## C. Train/evaluate the same student TCN

```bash
python -m ml.train_tcn_prototype \
  --data-dir "/Volumes/Neda 1TB/20260615/IBDH_004_FDG_IBDH_004_FDG_155716/TCN_parametric" \
  --output-dir "/Volumes/Neda 1TB/20260615/IBDH_004_FDG_IBDH_004_FDG_155716/TCN_parametric_results" \
  --epochs 30

python -m ml.evaluate_tcn_prototype \
  --data-dir "/Volumes/Neda 1TB/20260615/IBDH_004_FDG_IBDH_004_FDG_155716/TCN_parametric" \
  --checkpoint "/Volumes/Neda 1TB/20260615/IBDH_004_FDG_IBDH_004_FDG_155716/TCN_parametric_results/best_tcn_prototype.pt" \
  --output-dir "/Volumes/Neda 1TB/20260615/IBDH_004_FDG_IBDH_004_FDG_155716/TCN_parametric_results/evaluation"
```

```bash
python -m pytest tests/test_parametric_tcn_dataset.py -v
```

The fitted JSON may contain values derived from a participant; keep
the config and simulation outputs local if their data governance
requires that. Do not commit fitted subject-specific model values
to the public repository.
