# Bq/mL dose-referenced simulation (experimental)

This mode was added separately from the original simulation entry point to
preserve the existing baseline. The generated NPZ files use the same keys
as `ml.train_tcn_prototype`, but **activity units are Bq/mL, not kBq/mL**.

## Required local inputs

Two CSVs containing `TimeStart`, `TimeEnd` in **seconds** and `aorta`
in **Bq/mL**. They should be decay-corrected to the beginning of the respective
single-tracer reference scan. Keep these human-derived data files out of GitHub.

Reference administered activities, after residual:
- FDG: **4.91 mCi = 181.67 MBq**
- PBR28: **4.98 mCi = 184.26 MBq**

PBR28 aortic blood is **whole blood**, not metabolite-corrected parent plasma.
No kinetic parameters are being estimated from it. `tissue-approx` runs the
existing fixed 2TCM using blood as a *surrogate* input; this is not a
physiologically validated tissue model. `blood` mode separates aortic
signals directly and should not be mistaken for tissue training data.

## Run

```bash
python -m sim.generate_bqml_dataset \
  --fdg-csv '/private/FDG Aorta(1).csv' \
  --pbr-csv '/private/Aorta PBR(1).csv' \
  --output-dir datasets/bqml_v1 \
  --n-samples 1000 \
  --pbr-duration-min 65 \
  --fdg-offsets-min 0 5 10 \
  --signal-source tissue-approx \
  --effective-sensitivity-cps-per-bq 0.001
```

The default effective sensitivity is a **placeholder**, not a Panorama GS
voxel sensitivity. Tune it to measured reconstructed-image variability before
interpreting realism. Sampled counts are *effective Poisson counts* and are
not equivalent to actual sinogram detector-event counts.

The initial run uses a truncated PBR frame schedule ending at 65 minutes
to avoid extrapolating the 65-minute FDG IDIF. A 90-minute mixed acquisition
requires an empirically justified FDG tail model or longer measured data.

**Assumption:** each single-tracer decay-correction reference (scan start)
coincides with injection. If that is not true, introduce the measured scan
start/injection offsets before interpreting the first-pass curves.

On the common timeline, PBR is first at time zero, FDG starts at the sampled
offset. The code dose-scales the *decay-corrected* IDIFs, generates separate
tracer signals, applies exponential physical decay from each reference time,
integrates each over output frames, **sums the physically decaying components**,
and only then samples noise. It does *not* add decay-corrected curves as the
measured mixture.

`Y_phys_tavg_n0_d1`: noise-free, physically decaying FDG/PBR28 contributions.
`Y_bio_tavg_n0_d0`: decay-corrected contributions (approximation to within-frame DC).
`X_clean_tavg_n0_d1`: sum of physical contributions.
`X_measured_tavg_n1_d1`: noisy physical mixture.
`expected_counts` and `sampled_counts`: count-domain intermediate outputs.
All TAC arrays are in Bq/mL; timestamps in minutes.

## TCN compatibility

```bash
python -m ml.train_tcn_prototype \
  --data-dir datasets/bqml_v1 \
  --output-dir ml/experiments/bqml_v1 \
  --epochs 30
python -m ml.evaluate_tcn_prototype \
  --data-dir datasets/bqml_v1 \
  --checkpoint ml/experiments/bqml_v1/best_tcn_prototype.pt \
  --output-dir ml/experiments/bqml_v1/evaluation
```

The legacy evaluator labels plots and physical errors as kBq/mL; those labels
must be updated before using it for scientific Bq/mL figures. The checkpoint
normalization uses training data and is otherwise unit-agnostic.

## Validation limits

- Aortic voxel mixture is not ground-truth dual-tracer tissue separation.
- Spatial correlations, randoms, attenuation, scatter and reconstruction
  noise are not modeled.
- Frame-mean IDIF interpolation approximates the continuous decay-corrected
  curve; exact inversion of a scanner's frame correction depends on its
  reconstruction convention.
- Dose scaling assumes linear tracer kinetics and consistent normalization.
- Split-level independence does not imply independence of real subjects:
  all synthetic samples here derive from the same two measured IDIFs.
- Keep participant-derived reference CSVs private; output NPZs also derive
  from participant data and should not be committed to a public repository.

## Testing an externally fitted FDG brain TAC

Keep fitted, participant-derived parameters local rather than hard-coding them into
this public repository. To inspect a fitted FDG model without changing PBR28:

```bash
python -m sim.validate_bqml \
  --fdg-csv "/private/FDG Aorta(1).csv" \
  --pbr-csv "/private/Aorta PBR(1).csv" \
  --output-dir results/bqml_qc_fitted \
  --duration-min 60 \
  --fdg-offsets-min 0 5 10 \
  --signal-source tissue-approx \
  --fdg-kinetics 0.146969833218802 0.119575138861569 0.042108367298586 0 \
  --fdg-vb 0.0808523133568341 \
  --fdg-delay-sec 5.78091827383968
```

All kinetic rate constants are interpreted in minutes-based units; the fitting
code's `td` is interpreted as seconds here, with positive delay shifting the
FDG input later. **Confirm the original fitter's delay sign convention**, and
whether its `vb` mixes tissue and blood with `(1-vb)C_t + vb C_b` before
claiming model equivalence. The FDG aortic curve is used as a surrogate for
parent-plasma input, so tissue fits may differ from Kevin's PIM. These
parameters are a single fitted brain-region candidate, not a representative
distribution of biological parameters. The new flags also work with
`sim.generate_bqml_dataset`; the defaults preserve the original illustrative
kinetic model and zero vascular fraction/delay.
