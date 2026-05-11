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

Shared state is managed through `core/share_variables.py` using a `multiprocessing.Manager().Namespace()` wrapper.

Configuration is loaded from `config.toml` (supports comments).
`config.json` is no longer accepted by the control runtime.

## Startup Device Validation

- Device switches in `config.toml` (`"on"` / `"off"`) control whether each device is required at experiment start.
- When an enabled startup-critical device cannot be opened, experiment start is blocked.
- The failure reason is reported in both:
  - the main GUI warning/error area
  - the terminal output
- To continue without a disconnected device, set that device to `"off"` in `config.toml`.

## Email Notifications

When an experiment finishes, PyCCAPT can send the operator a summary email with the experiment's `parameters.txt`
and `apt.log` attached, and the PyCCAPT logo embedded in the body.

To enable email:

1. Copy the template `pyccapt/files/email_credentials.example.toml` (checked in, no secrets)
   to `pyccapt/files/email_credentials.toml` (gitignored).
2. Fill in `sender_email`, `password`, and — if you don't use Gmail — `smtp_server` / `smtp_port`. For Gmail, generate a
   16-character App Password at <https://myaccount.google.com/apppasswords> and paste it as the `password` value.
3. Optionally set `cc = ["lab-archive@example.com"]` to copy a permanent address on every notification.
4. In the main GUI, type the recipient address into the "Email" field of the Run page before starting the experiment.

If the credentials file is missing or malformed, the experiment still runs to completion; the failure is recorded in the
experiment's `apt.log` (`Email notification failed`) and the GUI session log. The legacy file `email_pass.txt` (one-line
plaintext password) is still accepted as a fallback for backwards compatibility, but the TOML form is preferred.

`email_credentials.toml` and `email_pass.txt` are explicitly listed in `.gitignore` so they cannot be committed
accidentally.

## Logging

The control package writes two layers of logs.

- **GUI session log** — `pyccapt/files/logs/gui/gui_<YYYY-MM-DD>.log`
    - Daily rotating file (5 MiB per file, up to 10 backups per day).
    - Captures everything emitted by the GUI process and the experiment subprocess.
    - Each record includes timestamp, level, logger name, source `filename:line`, and message.
    - Captures: `print()` output, Python `warnings`, uncaught exceptions (full traceback), startup banner with version /
      host / OS / Python / detected COM ports, configuration snapshot at start.
- **Per-experiment log** — `<experiment folder>/meta_data/apt.log`
    - Created when an experiment starts.
    - Self-contained: includes the experiment name, pulse mode, configured COM ports, device toggles, V-dc range,
      detection rate, super-user override state, and every `INFO` or higher event from the experiment loop (
      initialization, ramps, stop reasons, HDF5 finalisation, cleanup).
    - The same records also appear in the daily GUI log so the operator can scroll across multiple experiments
      chronologically.

Both layers are configured by `pyccapt/control/core/loggi.py`:

- `setup_application_logging(project_root)` is called from `gui_main.__main__` and
  from `apt_exp_control.run_experiment` (the experiment subprocess). It is idempotent.
- `logger_creator(script_name, variables, log_name, path)` attaches a per-experiment file handler.

To raise console verbosity, call `setup_application_logging(project_root, console_level=logging.DEBUG)`. The file always
records at DEBUG.

When debugging "the software was working yesterday but failed today," the GUI session log under `files/logs/gui/` is the
first place to look — it records what hardware was visible at startup, which devices the operator enabled, and any
uncaught exceptions with full stack traces.

## Data Structure

HDF5 groups and dataset semantics are documented in [DATA_STRUCTURE.md](DATA_STRUCTURE.md).

## Folder Responsibilities

- `apt/`: experiment orchestration and control loop
- `core/`: shared state, logging, HDF5 writing, runtime helpers
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

`electrode.toml` stores available electrode identifiers used for experiment metadata entry in the GUI.
The file is comment-friendly and user-editable. Example:

```toml
[electrodes]
names = [
    "NiC1",  # Nickel capillary
    "CuC1",
    "NC",    # Not categorized
]
```
