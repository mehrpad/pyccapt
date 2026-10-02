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

Reusable TOML experiment queues, compact queue editing, explicit sample mapping
and recorded starting settings are described in
[EXPERIMENT_PLANS.md](EXPERIMENT_PLANS.md). Copy
[experiment_plan.example.toml](../files/experiment_plan.example.toml), select
**Setup Parameters → TOML Plan → Load TOML**, and review all rows before Start.
The main GUI opens at 760 × 670 and shows the queue when TOML Plan is selected.
TextBox provides the single-run form. In TextBox mode, **Advanced settings…**
opens detection mode, control algorithm, refresh frequency, pulse-voltage limits
and K_p controls together. These are the same controls used by the main form:
edits apply immediately and existing run locks remain in effect. The Advanced
settings label and button sit below the target Detection Rate input on the left.
A separator below the Run Statistics detection rate precedes Electrode and
Flat Test. A second line separates these buttons from the bordered Auto Alignment box
containing alignment start voltage, voltage increment and Automatic Alignment.
Start sits above Stop in a separate group just above the bottom full-width
separator.

Laser command units, readback semantics and recording corrections are documented
in [LASER_MANUAL_AUDIT.md](LASER_MANUAL_AUDIT.md).

Laser-stage scanning, focus, tracking, GUI controls and required calibration are
documented in [LASER_ALIGNMENT.md](LASER_ALIGNMENT.md).

## Runtime Architecture

The automatic sample-alignment sequence, commissioning settings, detector
analysis and metadata are documented in [AUTOMATIC_ALIGNMENT.md](AUTOMATIC_ALIGNMENT.md).
The initial transfer/settling/error journal is copied into each dataset as
`meta_data/alignment_transfer.jsonl`; failed pre-start transfers retain their file
under `data/alignment_sequences/`. The sample search checks the saved position
and 16 broad probes derived from the ±50 µm envelope. Repeatable rate jumps and
dense hitmap regions trigger a second pass within ±15 µm, then fine XY within
±5 µm. Targets are inset by 0.2 µm; failed searches return to saved Z/XY and retry
at +100 V within the configured voltage/time limits. Fine Z approach
requires stable XY centring and a circular fit and is capped at 20 µm total from
the saved position.

The control application uses multiple processes:

- main GUI process (`gui_main.py`)
- experiment process (`apt/apt_exp_control.py`)
- detector process (Surface Concept, RoentDek, or DRS)
- optional sub-GUI processes (cameras, visualization)

Shared state is managed through `core/share_variables.py` using a `multiprocessing.Manager().Namespace()` wrapper.

Experiment lifecycle is explicit and observable through `variables.experiment_state`:
`idle -> initializing -> running -> stopping -> safe_off -> finalizing -> complete`.
Any unhandled failure transitions to `failed`, records `experiment_error`, requests detector shutdown, and attempts the
idempotent hardware safe-off path before publishing the completion event. New code should use
`apt/experiment_state.py` rather than inventing additional lifecycle flags.

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

For hardware-free development set `tdc_model = "Simulator"`. The simulator follows the same stop-event and ring-buffer
contract as the real detector backends, so startup, acquisition, finalization, and GUI behavior can be exercised without
vendor SDKs. Its deterministic controls are `simulator_seed`, `simulator_batch_size`, and `simulator_interval_s`;
`simulator_fail_after_batches` injects a worker crash for tests and must remain `0` in normal dry runs.

Every detector now implements the same lifecycle contract (`start`, `stop`, `join`, `health`, and completed chunk
streaming). The experiment worker consumes an immutable `RunConfig`, accepts typed stop commands, and emits typed status
plus exactly one completion acknowledgement. The state sequence is guarded as `IDLE -> INITIALIZING -> RUNNING ->
SAFE_OFF -> FINALIZING -> COMPLETE`, with failures transitioning to `FAILED` only after safe-off is attempted.

For a physical interlock, configure `safety_interlock_backend = "nidaq"`, `safety_estop_input_channel`, and optionally
`safety_watchdog_output_channel`. Software access overrides are appended and fsynced to
`meta_data/safety_overrides.jsonl`; an override never bypasses a physical E-stop. The status bar reports worker health,
queue depth, dropped plot records, chunk-write latency, heartbeat age, and the safe-state acknowledgement.

The main command (and the backwards-compatible `pyccapt-data` alias) provides operational checks and recovery:

```text
pyccapt validate-config pyccapt/config.toml
pyccapt validate-hdf5 path/to/experiment.h5
pyccapt recover-run path/to/chunks path/to/recovered.h5
```

Chunks are published by temp-write/fsync/atomic-rename and recorded in per-stream JSONL manifests with row counts,
shapes, dtypes, IDs, checksums, write latency, and completion state. Recovery never overwrites the source chunks,
quarantines inconsistent evidence instead of deleting it, and writes the destination transactionally.

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

All control windows use `gui/responsive.py` to fit their frame inside the current
monitor's available desktop area, including taskbar space. Existing opening sizes,
layout order, fonts, button sizes and colours are retained when they fit. Plot and
camera panels can contract to readable minimum sizes or expand with the window.
Below the layout's minimum size, scrollbars provide access to the complete layout
rather than scaling down controls. Monitor changes and work-area changes trigger
another fit; ordinary resizing retains the operator's chosen window size.
The automatic stage alignment monitor still opens at 660 × 350 logical pixels.

![Main GUI](https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/main_gui.png?raw=True)

Detailed sub-GUI snapshots:

Pumps/Vacuum groups cryo temperatures and their target control, the three venting
buttons, and the six Buffer/LL/CLL chamber/pre-vacuum LCDs in bordered boxes.
The six LCDs occupy two rows of three, retaining their 150 × 50 sizes and warning
colours. The combined Pumps/Vacuum and Gates window opens at 1280 × 640, and the
standalone Pumps/Vacuum window opens at 840 × 720. Vacuum history, Gates controls,
load-lock temperature/baking controls and error messages remain available;
smaller monitors scroll without hiding controls.
Venting is aligned beside the Gates diagram with a gap from the vacuum displays.
The Buffer Chamber Pre label sizes to its full text and stays on one line.

Stage Control opens at 880 × 220 with compact 64 × 28 position readouts and
speed-selector widths sized for the configured table. The nine mm/µm/nm readouts,
three speed presets and jog-distance labels remain visible alongside all jog
buttons, Home, Reference, STOP and Override Access. Header spacing, Z jog spacing
and layout margins are tighter; long error messages still wrap.

Cameras defaults to 900 × 700 with smaller margins and one row per exposure
slider and value. All six overview/detail views, three camera connections,
sample-position controls and five instrument monitors remain visible. The
connection rows show the serial and slot on one line, with the full model and
state in a tooltip. Bottom notifications disappear after five seconds; routine
refreshes do not restore an expired message.

Laser Control defaults to 980 × 650. Settings and three optical readouts sit above
the alignment controls beside the response plot; stage position readouts, speed
presets, jog buttons, Home, Reference, STOP and Override Access occupy the row
below. Both plot tabs, alignment settings and CLI/NKTPBus controls remain
available. Fields accommodate their maximum values and speed presets, and
connection and alignment messages still wrap. Alignment fields have six pixels
between rows and wider column gaps so they cannot overlap. Connection warnings
and temporary errors share a two-line message area; scroll or hover to read
longer messages. The window grows to fit the controls when fonts require it.

Visualization opens at 980 × 570. Hold DC, Set DC and the target voltage field
share one row, and the LED with Running/Stopped text sits in the top-right
corner alongside the FDM count. The four upper rectangular panels have
identical pixel dimensions at every window size. Compact voltage controls and
two rows of spectrum controls reduce the width needed without removing any
plot, calibration view, status indicator or input. Smaller screens use scrolling.
The following Cameras, Laser and Visualization screenshots show layouts without
connected hardware or acquired data.

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
