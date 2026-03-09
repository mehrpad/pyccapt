# Control Module

The `pyccapt.control` package provides instrument control, live monitoring, and experiment data acquisition workflows for open-source atom probe tomography systems.

![OXCART atom probe](../pyccapt/files/readme_images/oxcart.jpg)

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
- device toggles should use `enabled` / `disabled`
- legacy `on` / `off` values still work

Electrode labels used in the GUI are configured in `pyccapt/control/electrode.toml`.

Example:

```toml
[electrodes]
names = [
  "NiC1", # Nickel electrode
  "CuC1", # Copper electrode
  "NC",   # Not categorized
]
```

Electrode naming in the control workflow:

![Electrode labels](../pyccapt/files/readme_images/electrode.png)

Use the electrode list to match the naming shown in the GUI and your lab workflow. For example, `NiC1` refers to the nickel electrode and `CuC1` refers to the copper electrode.

## Startup Device Validation

- Enabled devices are validated when an experiment is started.
- If a required device cannot be opened, startup is blocked and the failure is reported in:
  - the main GUI warning area
  - terminal output
- `Access Override` now asks for confirmation before it bypasses those checks.

To proceed without a disconnected device in the normal path, set that device to `disabled` in `config.toml`.

## Data Output

Control-side HDF5 schema details are documented in [Control_DATA_STRUCTURE.md](Control_DATA_STRUCTURE.md).

Runtime logs are stored in:

- `pyccapt/files/logs/vacuum`
- `pyccapt/files/logs/baking/<timestamp>`

## GUI Overview

![Main GUI](../pyccapt/files/readme_images/main_gui.png)

The main window is the experiment entry point. Long error messages now use a smaller wrapped font so port and device warnings remain readable inside the GUI instead of being clipped.

Sub-GUI views:

- Gates: ![Gates GUI](../pyccapt/files/readme_images/gates_gui.png)
- Pumps/Vacuum: ![Pumps GUI](../pyccapt/files/readme_images/pumps_gui.png)
- Cameras: ![Cameras GUI](../pyccapt/files/readme_images/cameras_gui.png)
- Laser: ![Laser GUI](../pyccapt/files/readme_images/laser_gui.png)
- Stage: ![Stage GUI](../pyccapt/files/readme_images/stage_gui.png)
- Visualization: ![Visualization GUI](../pyccapt/files/readme_images/visualization_gui.png)
- Baking: ![Baking GUI](../pyccapt/files/readme_images/baking_gui.png)
