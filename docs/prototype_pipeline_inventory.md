# Dual-Tracer TCN Prototype: Reproducible Baseline and File Inventory

Inventory date: 2026-08-20

This document freezes the current working prototype before cleanup. It records
the official execution path, configuration, outputs, and the role and Git state
of the relevant files. It does not propose that any file be deleted yet.

## Official three-command pipeline

Run all commands from the repository root in the `dual_tracer` Conda
environment.

### 1. Generate the 1,000-sample dataset

```bash
python -m sim.generate_dataset \
  --output-dir datasets/dual_tracer \
  --n-samples 1000 \
  --dataset-seed 48 \
  --split-seed 49
```

This overwrites the dataset files in `datasets/dual_tracer/`. The configuration
uses an 80/10/10 split, PBR28 first and FDG second, FDG delays of 0, 5, or 10
minutes, independently sampled AIF scales from 0.85 to 1.15, and Poisson count
noise. The same configuration and seeds should reproduce the same samples and
split membership.

Primary outputs:

- `datasets/dual_tracer/train.npz` (800 samples)
- `datasets/dual_tracer/val.npz` (100 samples)
- `datasets/dual_tracer/test.npz` (100 samples)
- `datasets/dual_tracer/dataset_metadata.json`
- one metadata JSON file for each split

### 2. Train the causal mixture-consistent TCN

```bash
python -m ml.train_tcn_prototype \
  --data-dir datasets/dual_tracer \
  --output-dir ml/experiments/mixture_30ep \
  --epochs 30 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --hidden-channels 16 \
  --num-levels 4 \
  --kernel-size 3 \
  --dropout 0 \
  --mixture-weight 0.1 \
  --seed 123 \
  --device auto
```

This command uses only the training and validation splits for optimization and
checkpoint selection. It predicts two physical TAC channels in canonical order:
FDG, then PBR28.

Primary outputs:

- `ml/experiments/mixture_30ep/best_tcn_prototype.pt`
- `ml/experiments/mixture_30ep/history.json`

### 3. Evaluate the frozen checkpoint on the test split

```bash
python -m ml.evaluate_tcn_prototype \
  --checkpoint ml/experiments/mixture_30ep/best_tcn_prototype.pt \
  --data-dir datasets/dual_tracer \
  --output-dir ml/experiments/mixture_30ep/evaluation \
  --batch-size 32 \
  --device auto
```

This is the first command in the official path that evaluates the held-out test
set. It does not retrain or update model weights.

Primary outputs:

- `ml/experiments/mixture_30ep/evaluation/test_report.json`
- `ml/experiments/mixture_30ep/evaluation/test_predictions.npz`
- `ml/experiments/mixture_30ep/evaluation/test_prediction_examples.png`

## Optional validation and analysis commands

These are downstream checks, not additional training stages.

```bash
python -m scripts.audit_generated_dataset --data-dir datasets/dual_tracer
python -m ml.decay_correct_predictions
python -m ml.analyze_biological_test_predictions
python -m ml.plot_prototype_parameters
```

The decay-correction and biological-analysis commands depend on the prediction
file created by command 3.

## Scientific baseline

| Setting | Current value |
|---|---|
| Samples | 1,000 |
| Split | 800 train / 100 validation / 100 test |
| Dataset seed | 48 |
| Split seed | 49 |
| Injection order | PBR28 first, FDG second |
| FDG delays | 0, 5, 10 minutes |
| AIF scale range | 0.85–1.15 independently for each tracer |
| Kinetic parameters | Fixed per tracer |
| Noise | Poisson scanner counts |
| Saved target channel order | FDG, PBR28 |
| TCN input | One noisy physical mixture TAC |
| TCN output | Two physical TACs |
| TCN causality | Causal; no future frames used |
| Hidden channels / levels | 16 / 4 |
| Kernel / dilations | 3 / 1, 2, 4, 8 |
| Dropout | 0 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Epochs | 30 |
| Training seed | 123 |
| Loss | tracer MSE + 0.1 × mixture MSE |
| Normalization | divide all arrays by maximum absolute noisy training input |

## File-state terminology

- **Baseline file**: tracked by Git and unchanged relative to the current commit.
- **Modified baseline file**: tracked by Git but currently changed locally.
- **Prototype-created file**: currently untracked and created during the
  prototype work. Git does not yet record an author or history for it.
- **Generated artifact**: data, checkpoint, report, cache, or figure rather than
  source code.

Because the working changes are uncommitted, Git cannot reliably attribute each
modified line to a specific person. Attribution should be reviewed before the
first cleanup commit.

## Dataset-generation files

| File | Role | Required by current path? | Current Git state |
|---|---|---:|---|
| `sim/generate_dataset.py` | Dataset command, configuration, and global metadata | Yes | Prototype-created |
| `sim/frame_schedules.py` | Defines the 62 PET frame edges | Yes | Baseline file |
| `core/timegrid.py` | Internal time grid and frame properties | Yes | Modified baseline |
| `sim/sample_generator.py` | Samples scale, delay, and child seed; creates samples | Yes | Modified baseline |
| `sim/tac_simulator.py` | Biological TACs, decay, frame averaging, and scanner measurement | Yes | Modified baseline |
| `sim/dataset_builder.py` | Canonicalization, validation, deterministic splitting, and saving | Yes | Modified baseline |
| `core/protocol.py` | Stores injection delay | Yes | Baseline file |
| `core/noise.py` | Applies the scanner Poisson noise model | Yes | Baseline file |
| `utils/frame_average.py` | Converts internal-grid values to PET frame averages | Yes | Modified baseline |
| `kinetics/helpers/twotcm.py` | Simulates the two-tissue compartment model | Yes | Baseline file |
| `tracers/tracer_factory.py` | Constructs FDG or PBR28 objects | Yes | Baseline file |
| `tracers/base.py` | Base tracer interface | Yes | Baseline file |
| `tracers/fdg.py` | FDG AIF, half-life, and kinetic parameters | Yes | Baseline file |
| `tracers/pbr28.py` | PBR28 AIF, half-life, and kinetic parameters | Yes | Modified baseline |
| `tracers/aif.py` | Mathematical AIF definitions and scale multiplication | Yes | Baseline file |

## TCN training and evaluation files

| File | Role | Required by current path? | Current Git state |
|---|---|---:|---|
| `ml/tcn.py` | Causal TCN architecture | Yes | Modified baseline |
| `ml/train_tcn_prototype.py` | Data loading, normalization, loss, training, validation, checkpoints | Yes | Prototype-created |
| `ml/evaluate_tcn_prototype.py` | Frozen-checkpoint test evaluation and physical plots | Yes for evaluation | Prototype-created |
| `ml/decay_correct_predictions.py` | Converts physical predictions to biological TACs | Optional downstream | Prototype-created |
| `ml/analyze_biological_test_predictions.py` | All-test-sample plots and biological MSE analysis | Optional downstream | Prototype-created |

## Audit, visualization, and testing files

| File | Role | Changes simulation or model weights? | Current Git state |
|---|---|---:|---|
| `scripts/audit_generated_dataset.py` | Validates saved arrays and calculates dataset statistics | No | Prototype-created |
| `scripts/plot_audit_summary.py` | Renders audit tables and charts | No | Prototype-created |
| `ml/plot_prototype_parameters.py` | Renders the prototype parameter table | No | Prototype-created |
| `sim/plotters.py` | Shared simulation and audit plotting functions | No | Modified baseline |
| `scripts/smoke_test_dataset_generation.py` | Small end-to-end generation check | No | Prototype-created |
| `tests/test_dataset_builder.py` | Dataset building, splitting, and saving tests | No | Prototype-created/untracked |
| `tests/test_simulation_data_representation.py` | Simulation array semantics and shape tests | No | Prototype-created/untracked |

## Generated artifacts, not source code

| Location | Contents |
|---|---|
| `datasets/dual_tracer/` | NPZ splits, metadata, audit report, dataset figures |
| `ml/experiments/mixture_30ep/` | Checkpoint, history, predictions, reports, and figures |
| `ml/prototype_outputs/` | Older prototype checkpoint/history output |
| `**/__pycache__/` and `*.pyc` | Python bytecode caches |
| `.DS_Store` | macOS filesystem metadata |

Generated artifacts should not be confused with implementation files. The
current repository policy is to version the canonical prototype dataset and
model checkpoints in Git. Caches, `.DS_Store`, temporary outputs, and redundant
figures should remain excluded. Before substantially increasing dataset size,
revisit whether binary files should move to Git LFS or external artifact
storage.

## Present but not part of this prototype path

The following groups are not called by the official three commands:

- `ml/train.py`, `ml/dataset.py`, and `ml/datasets/`: unfinished or legacy ML
  path.
- `separation/methods/` and `separation/helpers/`: analytical separation
  algorithms, not the TCN prototype.
- `kinetics/builders/`: gamma/basis-library construction, not dataset generation.
- `sim/experiment_runner.py`, `sim/builders.py`, `sim/sweep.py`, and
  `sim/delay_scale_experiments.py`: older experiment and sweep infrastructure.
- Older `scripts/run_*`, `scripts/separate_*`, and `scripts/test_ml.py` files.
- `notebooks/` and `results/testing_separation_algs_refactored/`: explanatory
  notebooks and historical outputs.

There is one architectural complication: `sim/tac_simulator.py` contains both
the current forward-simulation code and older separation-related functions, so
it imports some separation and gamma-basis modules even though the official
dataset path does not call those functions. This should be reviewed before any
code is moved.

## Phase 1 review decisions

Before cleanup, confirm these points:

1. The three commands above are the official prototype pipeline.
2. The 1,000-sample dataset and `mixture_30ep` checkpoint are the baseline to
   preserve.
3. Training predicts physical TACs; biological TACs are produced only through
   downstream decay correction.
4. The unfinished `ml/` files and analytical separation code are out of scope
   for the prototype, but should not yet be deleted.
5. The canonical dataset and checkpoints will be deliberately versioned in Git;
   caches and incidental generated files will not.

No file should be renamed, moved, or deleted until these five points are
approved.
