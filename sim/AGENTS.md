# Dual-Tracer PET Dataset Generation

This file contains instructions for work inside the `sim/` folder related to simulated dataset generation.

## Scope

The goal is to build and maintain the simulated dual-tracer PET dataset pipeline.

Work should focus on:

- `sim/tac_simulator.py`
- `sim/sample_generator.py`
- `sim/dataset_builder.py`
- `sim/generate_dataset.py`
- dataset-generation tests

Do not add or modify TCN model or training code yet.

## Scientific behavior

The simulation uses FDG and PBR28 TACs.

The kinetic parameters are intentionally fixed and must not be randomized, swept, or changed.

### Fixed FDG parameters

- `K1 = 0.12`
- `k2 = 0.25`
- `k3 = 0.06`
- `k4 = 0.00`

### Fixed PBR28 parameters

- `K1 = 0.18`
- `k2 = 0.25`
- `k3 = 0.04`
- `k4 = 0.01`

Samples may vary through:

- FDG scale
- PBR28 scale
- tracer injection order
- injection delay
- Poisson noise realization

Preserve the existing PET simulation physics unless a change is explicitly requested.

## Dataset arrays

Let:

- `N` be the number of samples
- `T` be the number of PET frames

Save the following arrays.

### `X_measured_tavg_n1_d1`

Noisy mixed measured-domain TAC.

Shape: `[N, 1, T]`

### `X_clean_tavg_n0_d1`

Clean mixed measured-domain TAC before Poisson noise.

Shape: `[N, 1, T]`

It must satisfy:

```python
X_clean_tavg_n0_d1[:, 0, :] == (
    Y_phys_tavg_n0_d1[:, 0, :] + Y_phys_tavg_n0_d1[:, 1, :]
)
```

within an appropriate floating-point tolerance.

### `Y_phys_tavg_n0_d1`

Individual physical, radioactively decayed tracer TACs.

Shape: `[N, 2, T]`

Channel order must always be:

- channel `0`: FDG
- channel `1`: PBR28

### `Y_bio_tavg_n0_d0`

Individual biological, non-decayed tracer TACs.

Shape: `[N, 2, T]`

Channel order must always be:

- channel `0`: FDG
- channel `1`: PBR28

### `expected_counts`

Expected mixed scanner counts before the Poisson draw.

Shape: `[N, T]`

## Canonical channel order

Simulation-time tracer order may change, but saved output channels must always be canonicalized:

```text
channel 0 = FDG
channel 1 = PBR28
```

Do not assume tracer 1 is always PBR28 or that tracer 2 is always FDG.

## Sample representation

Each generated sample should retain:

- noisy measured mixture
- clean measured mixture
- tracer 1 biological TAC
- tracer 2 biological TAC
- tracer 1 physical TAC
- tracer 2 physical TAC
- expected counts
- sample metadata

All TACs stored in the final dataset must be frame-level arrays of shape `[T]`.

Internal-grid arrays may be retained for plotting or simulation internals, but
they must be clearly named with a `_tint` time-grid marker. Frame-averaged
arrays use `_tavg`; noise and decay states use `_n0`/`_n1` and `_d0`/`_d1`.

## Metadata

Per-sample metadata should include:

- sample ID
- tracer names
- tracer scales
- injection delay
- injection order or first tracer
- sample seed
- decay constants when useful

Global dataset metadata should include:

- number of samples
- split sizes
- frame edges
- frame midpoints
- frame durations
- internal time step
- scanner name
- voxel size or ROI setting
- noise model
- dataset seed
- channel definitions
- units
- fixed FDG kinetic parameters
- fixed PBR28 kinetic parameters

Convert NumPy scalar values to normal Python values before writing JSON.

## Reproducibility

Use a master dataset seed.

Each sample should receive its own stored child seed so that the sample can be regenerated independently.

The same configuration and seed must reproduce the same dataset.

## Splits

Support train, validation, and test splits.

Default fractions may be:

- train: `0.8`
- validation: `0.1`
- test: `0.1`

Verify that:

- every sample appears in exactly one split
- no split indices overlap
- all samples are assigned

If multiple noise realizations are later generated from the same clean simulation, keep all such realizations in the same split.

## Saving

Prefer one compressed NPZ file per split:

```text
train.npz
val.npz
test.npz
```

Each split file should contain:

- `X_measured_tavg_n1_d1`
- `X_clean_tavg_n0_d1`
- `Y_phys_tavg_n0_d1`
- `Y_bio_tavg_n0_d0`
- `expected_counts`

Save per-sample metadata as JSON and global configuration metadata separately.

Use `pathlib.Path`.

Create parent directories when needed.

Do not use hardcoded absolute paths.

Save neural-network arrays as `float32` unless there is a clear reason not to.

## Validation

Before saving, verify:

- all expected shapes are correct
- all samples use the same number of frames
- arrays contain no NaN values
- arrays contain no infinite values
- frame durations are positive
- `X_clean_tavg_n0_d1` equals the sum of the two physical tracer channels
- output channel order is always FDG then PBR28
- save/load round trips preserve arrays and metadata

## Tests

Add focused tests for:

- correct array shapes
- canonical FDG/PBR28 channel order
- clean mixture reconstruction
- deterministic output with the same seed
- changed output with a different seed
- split completeness and non-overlap
- save/load round trip
- JSON serialization of metadata

Run the repository's existing test command when available. Otherwise, use:

```bash
pytest
```

## Coding constraints

- Reuse existing simulation functions rather than duplicating simulation logic.
- Keep dataset generation separate from separation algorithms.
- Do not construct gamma libraries for dataset generation.
- Do not run NNLS or other separation methods during dataset generation.
- Do not modify unrelated files.
- Use concise type hints and docstrings.
- Remove large per-sample debug prints from production generation code.
- Raise clear errors for invalid shapes, invalid configuration, or unsupported tracer names.
