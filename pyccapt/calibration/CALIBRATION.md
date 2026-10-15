# Calibration

The `pyccapt.calibration` package provides data preparation, calibration,
reconstruction, and visualization utilities for APT datasets.

The separate [guided processing notebook](tutorials/jupyter_files/data_processing_guided.ipynb)
adds accepted-state steps, resumable sessions and human-reviewed peak proposals.
Raw hits need no mass column. Interactive cropping uses a rectangle on DC voltage
versus event sequence and a circle on the detector density map, with a separate
preview and acceptance step.
See the [workflow guide](../../docs/guided_workflow.md) for dependencies, review
actions, cancellation, storage and scientific limitations. The legacy processing
notebook remains a separate entry point.

## Scope

The package is organized into these folders:

- `core`: core calibration logic, validation, shared state, and plotting APIs
- `clustering`: clustering and isosurface utilities
- `data_tools`: loading, conversion, and cropping workflows
- `leap_tools`: POS/EPOS/RRNG readers, Cameca raw importers (RHIT/STR/HITS), and LEAP plotting helpers
- `mc`: mass-to-charge and time-of-flight helper functions
- `reconstructions`: 3D reconstruction and structure-analysis tools
- `tutorials`: notebooks and notebook helper modules

## Shared State Model

Calibration workflows use `Variables` from
`pyccapt.calibration.core.share_variables`.

- `Variables` is the mutable state container shared across calibration workflows.
- `SharedVariablesBase` provides common validation and path helpers.
- Validation/state issues should raise explicit calibration exceptions
  (`CalibrationInputError`, `CalibrationStateError`).
- `sync_from_data(...)` is the single synchronization boundary after loading or replacing a dataframe. It also publishes
  `has_detector_positions`, `has_reconstruction`, and `has_mass_spectrum`; unavailable coordinates remain `NaN` rather
  than being represented by plausible-looking zero values.
- At that boundary, `CalibrationDataset` owns schema version, units, aligned row count, finite masks, capability flags,
  and a stable source hash. Reusing a `Variables` instance for a new load replaces this object and resets derived state.
- New correction models implement `CalibrationModel.fit()`, `predict_factor()`, `valid_domain`, and JSON-serializable
  `provenance()` so a fitted result can be audited and reproduced.

## Naming Convention

Use canonical module names in all imports:

- `interactive_point_identification.py`
- `cloud_plotter.py`

## Cross-Platform Paths

Use `pathlib.Path` and shared filesystem helpers in
`pyccapt.calibration.path_utils`:

- `ensure_directory`
- `build_output_path`
- `save_figure`

These helpers are intended to work on both Windows and Linux.

## Development Guidelines

- Follow PEP 8 naming and formatting conventions.
- Prefer explicit exceptions over `print` for invalid inputs.
- Keep modules focused by responsibility.
- Add tests for behavior changes under `tests/`.
- Treat non-finite detector, time-of-flight, and mass values as unavailable input before histogramming, fitting, or spatial
  indexing.

## Module Length Guardrail

A test guardrail enforces a file-size ceiling for Python modules in calibration
folders (including tutorial helpers):

- maximum allowed: `1250` lines per `.py` file
- enforced by: `tests/calibration/test_calibration_module_lengths.py`

The numerical peak models, advanced calibration algorithms, schema boundary, and canonical voxel/isosurface operations
live in focused domain modules rather than plotting or notebook UI modules. Compatibility imports preserve existing
notebook APIs while letting the domain code be tested without constructing widgets.

## Correction mathematics and unavailable inputs

The legacy voltage, bowl, joint, spline, ML and reference fits estimate a dimensionless
factor in the **active calibration domain**. If the fit target is
`f = observed / reference`, application is `corrected = observed / f`, for both
mass-to-charge (Da/e, labelled Da in existing columns) and time-of-flight (ns).
Only a factor actually fitted in the squared-time/mass domain would require a
square root before application to time. Residual log-shifts fitted to an active
spectrum similarly apply as `corrected = observed * exp(shift)` in that domain.
Optional per-stage refinement uses the same factor convention as its initial fit.

The initial straight-flight normalization uses detector coordinates converted from
cm to mm and the supplied flight path `L` in mm. With
`D_i = L**2 + x_i**2 + y_i**2`, it applies
`t_i' = t_i * sqrt(mean(D_valid) / D_i) * sqrt(V_i / mean(V_valid))`.
This compensates the distance and voltage dependence of offset-corrected flight
time; it is not an absolute time-zero fit. An instrument time offset must be
handled by its corresponding physical calibration step. Unknown detector
coordinates receive an identity spatial factor. Non-finite/non-positive voltage
invalidates only that event's result; no finite positive voltages raises
`CalibrationInputError`. Unavailable coordinates are never replaced with invented
positions for fitting.

Joint ML/reference models train and predict on complete feature rows. Rows without
the required features retain their incoming calibrated values. This is a
capability boundary, not a claim that those events have received spatial correction.
Acceptance requires finite quality evidence when validation peaks exist. A failed
holdout measurement or missing reference quality cannot count as successful
validation. Preview operations must leave the source state unchanged.

The optional guided physical model and its units, validation roles and limits are
documented in [calibration mathematics](../../docs/calibration_math.md).
The optional reference-template isotope fit is described in
[isotope-family calibration](../../docs/calibration_isotope_family.md).

## Exact computation and bounded memory

- Voltage samples are assigned to the same half-open intervals once, retaining
  original order within each interval and an inclusive final endpoint.
- Adaptive residual likelihoods retain every ion and every candidate shift while
  evaluating shift columns in blocks (64 MiB target workspace per worker, with a
  minimum of two full ion vectors). Spatial aggregation and the objective are
  unchanged; floating-point reduction equivalence is checked with tight tolerances.
- Joint polynomial and ML application use batches of 250,000 events by default.
  This changes allocation size, not the trained model or selected events.
- Candidate screening retains scores instead of every full corrected array.
  Only finalists are reconstructed for the already-configured full-score stage.
- `ml_calibration(..., diagnostic_cv=True)` requests additional diagnostic models.
  Console verbosity does not enable this extra training work.
- HDF5 imports assemble typed dataframe columns directly instead of routing them
  through an object matrix. Native counters (including integers above `2**60`)
  retain their precision; the legacy object-matrix helper remains available.

Coarse-to-fine candidate pruning remains an explicit approximation: it can select
a different result from exhaustive full-score evaluation. The legacy preset
disables it; the displayed default preset is applied when its widget is created.
No cache is reused across mutable notebook datasets without a valid revision key.

Regression coverage includes known-truth mass/ToF drift recovery, straight-path
geometry, partial events, invalid voltages, finite validation gates, preview state
preservation, independent import order, preset state, voltage-bin edge equivalence,
blocked likelihoods and chunked model predictions. See
`tests/calibration/test_calibration_audit_fixes.py`,
`test_voltage_correction_mode_scaling.py` and `test_calibration_exact_performance.py`.

## Tutorials

The new `tutorials/jupyter_files/data_processing_guided.ipynb` provides a staged
Import → Inspect → Calibrate → Range → Reconstruct → Analyze → Export workflow.
The existing `data_processing.ipynb` remains available. See
[guided workflow](../../docs/guided_workflow.md) for review/acceptance and
[analysis sessions](../../docs/analysis_sessions.md) for resumable HDF5 snapshots,
event identity, learned models, NeXus mapping and storage tools.

Examples are available under:

- `tutorials/jupyter_files`
- `tutorials/colab`

## Range Files

Saved range tables can be loaded across the workflows from either:

- PyCCAPT HDF5 range files: `.h5`
- IVAS range files: `.rrng`
- Legacy LEAP/IVAS range files: `.rng`

Use `pyccapt.calibration.data_tools.data_tools.read_range(...)` for the
normalized PyCCAPT dataframe, or the low-level parsers
`pyccapt.calibration.leap_tools.leap_tools.read_rrng(..., return_tables=True)` and
`pyccapt.calibration.leap_tools.leap_tools.read_rng(..., return_tables=True)`
when you need the raw IVAS/LEAP `ions` and `ranges` tables.

## LEAP APT Import Notes

The LEAP `.apt` reader now preserves the `Position` section layout used by
APTSuite/paraprobe-style readers by skipping the leading tip-box bounds before
reading the ion coordinates.

