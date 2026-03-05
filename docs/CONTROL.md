# Control Module

The `pyccapt.control` package provides instrument control, live monitoring, and experiment data acquisition workflows for open-source atom probe tomography systems.

## Responsibilities

The control module is responsible for:

- experiment orchestration and control loops
- communication with detector and auxiliary hardware
- GUI-based runtime operation and monitoring
- synchronized shared state across cooperating processes
- writing experiment metadata and acquisition streams

Calibration and reconstruction are implemented in `pyccapt.calibration`.

## Runtime Architecture

The application runs as multiple processes, typically including:

- main GUI process
- experiment/control process
- detector backend process (for example, Surface Concept, RoentDek, or DRS)
- optional sub-GUI processes

Shared state is handled via `pyccapt/control/core/share_variables.py`.

## Configuration

Control runtime configuration is loaded from `pyccapt/config.toml`.

- supported format: TOML
- legacy `config.json` is not supported

Electrode labels used in the GUI are configured in `pyccapt/control/electrode.toml`.

Example:

```toml
[electrodes]
names = [
  "NiC1", # Nickel capillary
  "CuC1",
  "NC",   # Not categorized
]
```

## Startup Device Validation

- Device switches in `config.toml` (`"on"` / `"off"`) define whether a device is required.
- Startup-critical enabled devices are validated at experiment start.
- If a required device cannot be opened, startup is blocked and the failure is reported in:
  - the main GUI warning/error area
  - terminal output

To proceed without a disconnected device, set that device to `"off"` in `config.toml`.

## Data Output

Control-side HDF5 schema details are documented in [Control_DATA_STRUCTURE.md](Control_DATA_STRUCTURE.md).

## GUI Overview

![Main GUI](../pyccapt/files/readme_images/main_gui.png)

Sub-GUI views:

- Gates: ![Gates GUI](../pyccapt/files/readme_images/gates_gui.png)
- Pumps/Vacuum: ![Pumps GUI](../pyccapt/files/readme_images/pumps_gui.png)
- Cameras: ![Cameras GUI](../pyccapt/files/readme_images/cameras_gui.png)
- Laser: ![Laser GUI](../pyccapt/files/readme_images/laser_gui.png)
- Stage: ![Stage GUI](../pyccapt/files/readme_images/stage_gui.png)
- Visualization: ![Visualization GUI](../pyccapt/files/readme_images/visualization_gui.png)
- Baking: ![Baking GUI](../pyccapt/files/readme_images/baking_gui.png)
