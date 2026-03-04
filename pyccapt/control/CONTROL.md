# PyCCAPT Control Module

<img align="right" src="https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/logo2.png" alt="PyCCAPT logo" width="100" height="100">

The `pyccapt.control` package drives instrument control, device communication, live monitoring, and experiment data acquisition for open-source atom probe tomography systems.

## Scope

This module is responsible for:

- instrument control loops (high voltage, pulse/laser settings)
- hardware integration (TDC, DRS, pumps, gauges, stage, cameras)
- GUI operation (main and sub-GUIs)
- synchronized multi-process shared state
- writing experiment metadata and raw data

Calibration and reconstruction are implemented in `pyccapt.calibration`.

## Runtime Architecture

The control application uses multiple processes:

- main GUI process (`gui_main.py`)
- experiment process (`apt/apt_exp_control.py`)
- detector process (Surface Concept, RoentDek, or DRS)
- optional sub-GUI processes (cameras, visualization)

Shared state is managed through `control/share_variables.py` using a `multiprocessing.Manager().Namespace()` wrapper.

Configuration is loaded from `config.toml` (supports comments).
`config.json` is no longer accepted by the control runtime.

## Data Structure

HDF5 groups and dataset semantics are documented in [DATA_STRUCTURE.md](DATA_STRUCTURE.md).

## Folder Responsibilities

- `apt/`: experiment orchestration and control loop
- `control/`: shared state, logging, HDF5 writing, runtime helpers
- `devices/`: hardware-specific device interfaces and initialization
- `devices_test/`: standalone per-device diagnostic scripts
- `drs/`: DRS digitizer wrapper and native libraries
- `gui/`: main GUI and sub-GUIs
- `nkt_photonics/`: NKT Origami interfaces
- `tdc_roentdek/`: RoentDek TDC wrapper and native libraries
- `tdc_surface_concept/`: Surface Concept TDC wrapper and native libraries
- `thorlabs_apt/`: Thorlabs stage control wrappers
- `usb_switch/`: USB switch wrapper

## Notes for Developers

- Use `pathlib.Path` or `runtime.project_path(...)` for portable paths.
- Keep hardware-facing logic isolated in `devices/`, `tdc_*`, `drs/` modules.
- Keep GUI logic in `gui/` and avoid direct hardware access from UI classes.
- Use `devices_test/` scripts to validate each device independently before full experiment runs.

## GUI Overview

![Main GUI](https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/main_gui.png?raw=True)

Detailed sub-GUI snapshots:

- Gates: ![Gates GUI](https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/gates_gui.png?raw=True)
- Pumps/Vacuum: ![Pumps GUI](https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/pumps_gui.png?raw=True)
- Cameras: ![Cameras GUI](https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/cameras_gui.png?raw=True)
- Laser: ![Laser GUI](https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/laser_gui.png?raw=True)
- Stage: ![Stage GUI](https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/stage_gui.png?raw=True)
- Visualization: ![Visualization GUI](https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/visualization_gui.png?raw=True)
- Baking: ![Baking GUI](https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/baking_gui.png?raw=True)

## Electrode List

`electrode.json` stores available electrode identifiers used for experiment metadata entry in the GUI.
