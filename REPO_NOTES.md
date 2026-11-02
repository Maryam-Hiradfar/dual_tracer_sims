# Dual-tracer PET walkthrough notes

Session documentation begins 2026-11-02. Explanations below come from source
inspection unless a runtime check is explicitly recorded. User has confirmed
simulation, dataset assembly, splitting, and saving make sense; ML walkthrough
remains pending.

## Pipeline and file-purpose map

- Simulation entry: `sim/generate_dataset.py`; configuration, time grid, metadata.
- Sample creation: `sim/sample_generator.py`; sampled scales/delays and child seeds.
- Forward simulation: `sim/tac_simulator.py`; kinetics, decay, frame averaging, noise.
- Physics: `tracers/{fdg,pbr28,aif,tracer_factory}.py`,
  `kinetics/models/twotcm.py`, `core/{timegrid,protocol,noise}.py`,
  `sim/frame_schedules.py`, `utils/frame_average.py`.
- Dataset assembly/splits/storage: `sim/dataset_builder.py` (not yet walked through).
- Preprocessing/training/loss: `ml/train_tcn_prototype.py`; model: `ml/tcn.py`.
- Evaluation: `ml/evaluate_tcn_prototype.py`; downstream decay correction,
  biological analysis, difficulty analysis, and plotting scripts in `ml/`.
- Audits/checks: `scripts/audit_generated_dataset.py`,
  `scripts/smoke_test_dataset_generation.py`, `tests/`.
- Existing inventory/run commands: `docs/prototype_pipeline_inventory.md`;
  its historical file-state table is not the current Git state.
- Other paths: analytical `separation/`, gamma builders, older scripts and
  unfinished ML loaders are outside the primary prototype path.

## Learning notes by component

### Simulation — source reviewed; user understanding confirmed

- `generate_and_save_dataset` configures PBR28 first, FDG second, delays
  0/5/10 min, independent uniform AIF scales 0.85–1.15, and no randomized order.
  Fixed kinetics restrict this baseline to a narrow family of synthetic curves.
- `DatasetGenerator.sample_spec` draws scales/delay and a child seed;
  `make_sample_generator` builds tracers and a per-sample noise RNG.
  `SampleGenerator.generate_sample` retains seven frame-level arrays plus metadata.
- `default_frame_edges` defines 62 frames over 87 2/3 min: 14×5 s,
  3×30 s, 5×60 s, 40×120 s. `TimeGrid` uses a 1 s internal grid.
  Internal arrays have shape `[M]`; frame arrays `[62]`.
- AIFs are prescribed plasma input curves, nominally kBq/mL, scaled before
  tissue simulation. FDG uses Feng parameters with a 0.27 min onset offset;
  PBR28 uses a sum of three exponentials. Scale is dimensionless, not an
  explicit injected-dose parameter. AIF empirical validity is not verified here.
- `kinetics/models/twotcm.py:simulate_2tcm` uses forward Euler from zero tissue
  concentrations: dC1/dt=K1*Cp-(k2+k3)*C1+k4*C2;
  dC2/dt=k3*C1-k4*C2; Ct=C1+C2. K1 is mL plasma/(mL tissue·min),
  k2–k4 are min^-1. FDG rates are (0.12,0.25,0.06,0), PBR28
  (0.18,0.25,0.04,0.01). No explicit vascular blood-volume term is used.
- Physical Ct is biological Ct times exp(-ln(2)*time_since_injection/half_life).
  Half-lives in the factory: FDG 109.8 min, PBR28 20.4 min. Decay is applied
  on the internal grid before frame averaging; average(Ct*decay) generally
  differs from average(Ct)*average(decay). `_tint` is used for computation,
  not only smooth plotting despite some source comments.
- `frame_average` integrates a piecewise-linear curve with interpolated frame
  boundaries and divides by frame duration. The clean mixture is the sum of
  physical frame curves. Noise uses a 4 mm cube (0.064 mL), sensitivity
  preset 176.3 cps/kBq, expected counts=Ct*V*S*duration_seconds, then a
  Poisson draw and conversion back to kBq/mL. This is an idealized independent
  count model, without explicit reconstruction/spatial effects.
- Open concern: delayed PBR28 uses max(t-delay,0), but its AIF at zero is
  positive, yielding pre-injection activity when PBR28 is second. Canonical
  PBR28-first generation avoids this path. Existing reversed-order tests do
  not establish physical timing correctness. No fix proposed for application yet.

### Dataset assembly — source reviewed; user understanding confirmed

- `stack_samples(samples, specs)` consumes matching nonempty lists of generated
  data and recipes; it does not simulate additional data.
- `canonicalize_sample` identifies tracers by name and orders both biological
  and physical targets as FDG then PBR28, independently of injection order.
- Mixtures and expected counts are retained; recipe fields enrich metadata.
  Stacking adds sequential `sample_id` values.
- Each source curve must be one-dimensional and have the same frame length.
  Stacked arrays must be finite, and clean mixture must equal physical target sum.
- `np.stack` gives mixtures `[N,T]`; `[:,None,:]` inserts the one input channel,
  yielding `[N,1,T]`. Stacking two targets per sample gives `[N,2,T]`.
- All five output arrays are float32; expected counts are `[N,T]`.
  Concentration units remain kBq/mL; normalization has not happened here.
- Returns five arrays and metadata as a tuple, passed onward to `split_dataset`.
  Dataset-builder tests have not yet been run this session; splitting and saving
  walkthroughs remain pending.

### Splitting and saving — source reviewed

- User confirmed the index-based split explanation, nested dictionary
  structure, and saving explanation.
- `split_dataset` validates nonnegative fractions summing to one and matching
  sample counts across arrays/metadata. A seeded permutation chooses membership.
- Train/validation sizes use integer truncation; test gets the remainder.
  Very small datasets may have empty splits. Canonical 1000 samples yield
  800/100/100. The same sample indices select every array and metadata entry.
- `_validate_split_indices` checks complete, disjoint assignment of sample IDs.
  This does not ensure disjoint kinetic shapes or delay categories across splits.
- Result is a nested dict keyed by train/val/test; each contains five arrays,
  original `sample_indices`, and a list of sample metadata dictionaries.
- `save_dataset_splits` creates the output directory, requires all three splits
  and five array keys, and writes compressed NPZ arrays plus int64 sample indices.
- Per-split JSON stores sample metadata; `dataset_metadata.json` stores global
  configuration plus computed split sizes. `_json_safe`
  recursively converts NumPy values/arrays and Paths into JSON-compatible values.
- Existing filenames are overwritten. No canonical data regenerated this session.
  Saving does not perform normalization or training; save/load behavior has not
  yet been tested this session.

### Preprocessing, TCN, training, evaluation

Preprocessing and batching explained; user requested proceeding to model.
User confirmed loss/training-loop explanation makes sense. Evaluation walkthrough
started; understanding remains unconfirmed.

#### TCN — source reviewed without implementation changes

- Trainer constructs `Model(**model_config).to(device)` from `ml/tcn.py`.
  Defaults: 1 input channel, 16 hidden channels, 4 blocks, kernel 3, 2 outputs.
- `Model.forward` maps `[B,1,T]` through four temporal blocks to `[B,16,T]`,
  then a 1x1 Conv1d projects to `[B,2,T]`. Output channels mean FDG/PBR28
  because training targets use that order; no biological curves are predicted.
- Each `TemporalBlock` contains two Conv1d→Chomp→ReLU→Dropout sequences,
  adds an identity or 1x1-projected residual input, then applies ReLU.
- Dilations are 1,2,4,8. Kernel 3 at dilation d reads offsets 0,d,2d in
  a causal convolution, expanding temporal context without longer kernels.
- Padding=(kernel_size-1)*dilation plus right-side cropping in `Chomp1d`
  preserves length and prevents future-frame access by construction.
- Default receptive field is 1+2*(3-1)*(1+2+4+8)=61 frames including current.
  Early outputs partly use zero padding; the last of 62 frames can use frames
  1 through 61 (zero-based), not frame 0.
- Convolution operates over frame indices, which have unequal durations;
  model receives no explicit frame times, durations, scales, or injection delay.
- Final projection is linear with bias, allowing negative predictions; mixture
  equality is not enforced by architecture. Dropout defaults to zero. Shape and
  causality are source-derived here; runtime verification remains pending.

#### Preprocessing — reviewed without implementation changes

- Training entry is `python -m ml.train_tcn_prototype`, calling `main()`.
  It parses CLI settings, checks basic ranges, seeds RNGs and chooses a device.
- `training_scale` computes a single maximum absolute value over all noisy
  training inputs, not per sample or channel. It must be finite and positive.
- `DualTracerDataset` loads noisy mixture x `[N,1,T]`, physical ground truth y
  `[N,2,T]`, and clean mixture `[N,1,T]` from NPZ as float32 PyTorch tensors.
- It validates shapes and divides all three tensors by the same training scale.
  Arrays become dimensionless; relative amplitudes and additive mixture relation
  are preserved. Validation/test values are not guaranteed to be <=1.
- Biological targets, counts and metadata are not loaded by this class.
  `__getitem__` yields one (x,y,x_clean) tuple; `__len__` gives sample count.
- DataLoaders batch into `[B,1,T]`, `[B,2,T]`, `[B,1,T]`; training shuffles,
  validation does not. Default batch size is 32; final batch may be smaller.
- `main()` loads the test split eagerly but test metrics require the explicit
  evaluation option. Optional overfit mode intentionally reuses training samples
  for validation; it is a debugging mode rather than held-out validation.
- Defaults differ from documented baseline: 20 epochs and mixture weight 0
  versus baseline CLI's 30 epochs and weight 0.1. No training launched here;
  normalization/loader behavior explained from source, not runtime-verified.

#### Loss and training — source reviewed; user understanding confirmed

- `batch_loss` averages squared prediction errors over all batch/channel/frame
  entries. Targets are normalized physical curves, not biological curves.
- Predicted mixture sums the two output channels with `keepdim=True`, giving
  `[B,1,T]`, and compares against clean mixture rather than noisy input.
- Total loss = tracer MSE + mixture_weight * mixture MSE. Baseline weight 0.1
  is a soft consistency penalty, not an exact constraint; default weight is 0.
  Correct sum alone cannot establish correct individual tracer separation.
- MSE weights frames equally despite unequal durations and count uncertainty.
  Loss values are in normalized units; concentration MSE equals normalized MSE
  times scale squared. No duration/count weighting is present.
- `run_epoch` enables training when an optimizer is provided. For each batch:
  clear gradients, predict, compute loss, backpropagate, then Adam updates weights.
- Validation calls the same function with optimizer=None: eval mode, gradients
  disabled, no parameter updates. Metrics weight batch means by sample count,
  handling a smaller final batch correctly for these fixed-length samples.
- After every epoch, lower total validation loss replaces the best checkpoint.
  Training runs all requested epochs (no early stopping); history records all
  epochs. Checkpoint holds weights, optimizer state, epoch, scale, configs,
  validation loss, target name and channel order.
- Optional test evaluation reloads best weights after training. Canonical separate
  evaluation script remains to be reviewed. No training/runtime loss check run
  this session; reviewed code unchanged.

#### Evaluation — source reviewed; user understanding unconfirmed

- `ml/evaluate_tcn_prototype.py:main` calls `evaluate` on a saved checkpoint
  and test split; no optimizer or training updates are involved.
- Rebuilds Model using checkpoint configuration, restores weights, and reuses
  saved training normalization scale rather than deriving a test-specific scale.
- Uses `model.eval()` and `torch.no_grad()`; unshuffled batches preserve test
  order. Concatenated predictions have `[N_test,2,T]` shape.
- Multiplies predictions, targets and mixtures by scale to restore kBq/mL.
  This restores units; radioactive decay is still present (not decay correction).
- Computes MSE/RMSE/MAE per tracer and overall tracer MSE; mixture MSE compares
  predicted channel sum against clean mixture. Reports normalized and physical
  metrics, with equal sample/frame weighting and no uncertainty intervals.
- Saves `test_predictions.npz`, `test_report.json`, and example PNG. Examples
  select one sample per delay near balanced scales; metrics cover all test samples.
- Model predicts physical targets; biological conversion is downstream and
  remains to be traced. Existing evaluation artifacts have not been rerun or
  verified this session; no results claims inferred from source inspection.

User subsequently confirmed evaluation makes sense.

#### Biological correction and amplification — source reviewed; unconfirmed

- `ml/decay_correct_predictions.py` consumes physical prediction NPZ, test
  biological ground truth, timing metadata and per-sample decay constants.
- `frame_decay_factors` averages exp(-lambda*time_since_injection) on the
  internal grid. Correction divides each physical prediction by its factor;
  arrays remain `[N,2,T]` in kBq/mL, now estimating biological curves.
- Only PBR28-first/FDG-second metadata is supported explicitly; factors are
  assigned in canonical FDG/PBR28 channel order, with FDG injection delay.
- Inverse decay amplifies physical prediction errors increasingly over time,
  particularly for shorter-half-life PBR28. Pointwise error multiplier is 1/d.
- Frame-average correction is approximate: avg(C*d)/avg(d) need not equal
  avg(C). Script measures a correction floor by correcting true physical
  targets and comparing to true biological targets.
- Exact decomposition: biological error = physical error/d + correction-floor
  residual. Therefore observed RMSE ratios need not equal a single inverse
  factor and include both prediction error and frame-averaging effects.
- Saves biological predictions, ground truth and factors in NPZ plus reports
  and plots. `ml/plot_decay_error_amplification.py` compares physical/biological
  errors by time region and selects a representative sample near median late
  error ratio. Numerical artifacts have not been verified this session.

User confirmed biological correction/amplification makes sense. Saved summary
metrics were inspected in `evaluation/test_report.json` and
`decay_correction/biological_test_report.json`; these are existing artifacts,
not a fresh evaluation. JSON can be loaded with `json.loads(Path(...).read_text())`
and printed with `pprint` or formatted per-tracer rows from `by_tracer`.

- `ml/tcn.py` defines architecture; `ml/train_tcn_prototype.py` uses it and
  supplies its own dataset loader, normalization, loss, training and checkpoints.
- `ml/train.py` is an unfinished older script (empty function bodies and invalid
  main guard); `ml/dataset.py` is empty. Neither is used by the canonical trainer.
- Git records earlier `tcn.py` work under MaryamH-MGB, and its prototype
  modification plus addition of `train_tcn_prototype.py` in Kayoung Kim's commit
  `bb26824`. Git author metadata does not establish whether Codex wrote lines;
  earlier assistant-session provenance is unavailable in this conversation.

### Simulation follow-up clarifications

- `samples` contains actual `HybridSample` results; `specs` contains their
  recipes. `make_sample_generator(spec)` constructs configured tracer,
  protocol, and RNG objects; the subsequent `generate_sample()` computes curves.
- Static reference search found no caller of `simulate_dual_tac_legacy` in
  repository source/notebooks. `simulate_dual_tac_any_alg` is called by
  `sim/sweep.py`. Neither is part of canonical dataset generation, which calls
  `simulate_dual_tracer_measurement` directly. External callers are not checked.

## Session change log

- Added `REPO_NOTES.md` to document the pipeline, simulation explanation,
  open concern, and checks. No implementation files edited this session.
- Existing work preserved. At this session's inspection `sim/generate_dataset.py`
  also had a pre-existing docstring change relative to the prior inventory turn.
- Updated `REPO_NOTES.md` with generator/analytical-path clarifications and
  dataset channel-ordering/stacking notes; implementation remains unchanged.
- Updated `REPO_NOTES.md` with split confirmation, nested-dictionary structure,
  saving behavior and overwrite semantics; implementation remains unchanged.
- Updated `REPO_NOTES.md` with user-confirmed simulation/data understanding and
  ML file roles/history; no implementation edits.
- Updated `REPO_NOTES.md` with preprocessing, normalization and loader notes;
  no implementation edits or training runs.
- Updated `REPO_NOTES.md` with TCN architecture, shape flow, causality and
  receptive-field notes; no implementation edits or training runs.
- Updated `REPO_NOTES.md` with loss, gradient updates, validation and checkpoint
  selection notes; no implementation edits or training runs.
- Updated `REPO_NOTES.md` with training confirmation and evaluation walkthrough
  notes; no implementation edits or evaluation runs.
- Updated `REPO_NOTES.md` with evaluation confirmation and downstream biological
  correction/amplification explanation; no implementation edits or new runs.
- User requested committing and pushing all current work. This includes existing
  tracked changes and untracked source, tests, instructions, analysis artifacts,
  figures and notes; ignored files remain excluded. User's additional on-disk
  `sim/tac_simulator.py` edits are preserved. Unsaved editor buffers cannot be
  captured by filesystem Git operations. Destination: origin/main in user's fork.

## Checks performed

- Read repository simulation instructions and inspected `git status --short --branch`:
  on `main`, with pre-existing tracked edits and untracked files. No reset,
  checkout, regeneration of canonical artifacts, push, or merge performed.
- `conda run -n dual_tracer python -m pytest tests/test_simulation_data_representation.py -q`:
  13 passed in 10.17 s. Covers representation, seeds, ordering, AIF overrides,
  frame integration, invalid grids, and decay-correction factors; passing tests
  do not establish empirical validity or pre-injection timing correctness.
- Numerical convergence, empirical AIF/scanner validity, and isolated
  end-to-end run remain unchecked. No expensive training run launched.
- Pre-commit `git diff --check` passed.
  `conda run -n dual_tracer python -m pytest tests -q`: 22 passed in 29.72 s.
  Isolated end-to-end run remains pending; current request is to save/push work.
