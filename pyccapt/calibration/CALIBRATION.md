# Calibration

The `pyccapt.calibration` package provides data preparation, calibration,
reconstruction, and visualization utilities for APT datasets.

## Scope

The package is organized into these folders:

- `core`: core calibration logic, validation, shared state, and plotting APIs
- `clustering`: clustering and isosurface utilities
- `data_tools`: loading, conversion, and cropping workflows
- `leap_tools`: POS/EPOS/RRNG readers and LEAP plotting helpers
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

