# Calibration

The `pyccapt.calibration` package provides data preparation, calibration,
reconstruction, and visualization utilities for APT datasets.

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

## Module Length Guardrail

A test guardrail enforces a file-size ceiling for Python modules in calibration
folders (including tutorial helpers):

- maximum allowed: `1250` lines per `.py` file
- enforced by: `tests/test_calibration_module_lengths.py`

## Tutorials

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

