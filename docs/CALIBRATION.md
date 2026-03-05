# Calibration Module

The `pyccapt.calibration` package provides workflows for atom probe tomography data preparation, calibration, reconstruction, and visualization.

## Core Workflows

Typical calibration workflows include:

1. Import and crop datasets (HDF5, EPOS, POS, ATO, CSV).
2. Correct time-of-flight and estimate `t0`/flight-path parameters.
3. Convert time-of-flight to mass-to-charge (`m/c`).
4. Apply voltage and bowl corrections.
5. Perform 3D reconstruction.
6. Define and apply ranging windows.
7. Generate 2D/3D visualizations and analysis plots.

## Package Structure

- `core`: validation, shared state, and primary calibration logic
- `data_tools`: loading, conversion, and preprocessing utilities
- `mc`: mass-to-charge and time-of-flight helper functions
- `reconstructions`: reconstruction and structural analysis tools
- `clustering`: clustering and isosurface workflows
- `leap_tools`: LEAP/POS/EPOS/RRNG import and helper tools
- `tutorials`: notebooks and notebook helper modules

## Shared State and Validation

Calibration workflows use shared mutable state through `Variables` in `pyccapt.calibration.core.share_variables`.
Validation and state-related errors should raise explicit calibration exceptions.

## Cross-Platform Paths

Use `pyccapt.calibration.path_utils` helpers for output and figure paths:

- `ensure_directory`
- `build_output_path`
- `save_figure`

## Data Structures

Calibration and range-file schema details are documented in [Calibration_DATA_STRUCTURE.md](Calibration_DATA_STRUCTURE.md).

## Tutorials

Interactive examples are available in:

- `pyccapt/calibration/tutorials/jupyter_files`
- `pyccapt/calibration/tutorials/colab`

Related user-facing tutorial pages are listed under [tutorials](tutorials.rst) in this documentation set.
