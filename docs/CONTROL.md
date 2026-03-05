# Control


PyCCAPT provides Python-based control software for atom probe tomography instruments. It includes
data acquisition workflows and device integration for systems such as Surface Concept and RoentDek
time-to-digital converter (TDC) hardware.

The control module is designed for experiment execution, live monitoring, and structured data capture.
Its modular structure supports extension to additional devices and control backends.

## Editable electrode configuration

The electrode list used by the control GUI is stored in:

- `pyccapt/control/electrode.toml`

This file is TOML-based and comment-friendly, so users can directly edit labels and keep notes:

```toml
[electrodes]
names = [
  "NiC1", # Nickel capillary
  "CuC1",
  "NC",   # Not categorized
]
```

## Startup device validation

- Devices can be enabled or disabled in `config.toml` with `"on"` / `"off"` switches.
- At experiment start, enabled startup-critical devices are checked.
- If a required device is not reachable, experiment start is stopped and the reason is shown in:
  - the main GUI warning/error area
  - terminal output
- If a device is intentionally disconnected, set it to `"off"` in `config.toml`.

## Main Control GUI Overview
![plot](../pyccapt/files/readme_images/main_gui.png)

The following sections show the primary control sub-GUIs.

## Gates Control GUI
![plot](../pyccapt/files/readme_images/gates_gui.png)

## Pumps, Vacuum, and Temperature GUI
![plot](../pyccapt/files/readme_images/pumps_gui.png)

## Cameras Control GUI
![plot](../pyccapt/files/readme_images/cameras_gui.png)

## Laser Control GUI
![plot](../pyccapt/files/readme_images/laser_gui.png)

## Stage Control GUI
![plot](../pyccapt/files/readme_images/stage_gui.png)

## Visualization GUI
![plot](../pyccapt/files/readme_images/visualization_gui.png)

## Baking Process GUI
![plot](../pyccapt/files/readme_images/baking_gui.png)








