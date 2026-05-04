# AGENTS.md

## Mission-Critical Orientation
- PyCCAPT has two primary domains: `pyccapt/control` (instrument runtime + GUI) and `pyccapt/calibration` (offline processing, calibration, reconstruction).
- Console entrypoint is `pyccapt=pyccapt.control.__main__:main` (`setup.py`), so most "app startup" changes begin in `pyccapt/control/__main__.py`.
- Control runtime is configuration-driven from `pyccapt/config.toml`; JSON config is intentionally rejected (`pyccapt/control/core/read_files.py`).
- HDF5 is the shared boundary format between acquisition and downstream workflows (`pyccapt/control/DATA_STRUCTURE.md`, `pyccapt/calibration/DATA_STRUCTURE.md`).

## Architecture You Need To Know First
- Control boot flow: `runtime.load_project_config()` -> `runtime.create_shared_context()` -> `gui_main.MyPyCCAPT(...)` (`pyccapt/control/__main__.py`).
- Cross-process state in control uses `multiprocessing.Manager().Namespace()` wrapped by `Variables` (`pyccapt/control/core/share_variables.py`); avoid adding ad-hoc globals.
- Detector backend selection is dynamic in `pyccapt/control/apt/detector_runtime.py` via `conf['tdc_model']` (`Surface_Consept`, `RoentDek`, `HSD`) and `variables.counter_source`.
- Calibration state is centralized in `pyccapt.calibration.core.share_variables.Variables`; validation failures should raise `CalibrationInputError` / `CalibrationStateError`.

## Non-Obvious Project Conventions
- Device toggles in `config.toml` may be `enabled/disabled` or `on/off`; loader normalizes to legacy `on/off` (`normalize_control_config`).
- Keep hardware I/O in `pyccapt/control/devices`, `pyccapt/control/tdc_*`, and `pyccapt/control/drs`; keep GUI classes in `pyccapt/control/gui`.
- Use path helpers instead of string-concatenated paths: `runtime.project_path(...)` (control), `pyccapt.calibration.path_utils` (calibration).
- Calibration modules have a hard size guardrail: max `1250` lines per `.py` file (`tests/calibration/test_calibration_module_lengths.py`).

## Developer Workflows (Repo Root)
```bash
pip install -e ".[full]"
pytest -q --run-control
pytest -q --run-calibration
pytest -q
sphinx-build -b html docs docs/_build/html
```
- `tests/conftest.py` auto-skips control/calibration suites when optional sentinel deps are missing; use explicit `--run-control` / `--run-calibration` when targeting one domain.
- For hardware troubleshooting, prefer targeted scripts in `pyccapt/control/devices_test/` before editing experiment orchestration.

## Integration Points And External Dependencies
- Control stack integrates with PyQt6 GUI, serial devices (`pyserial`), VISA instruments (`pyvisa`), detector SDK wrappers (`tdc_surface_concept`, `tdc_roentdek`, `drs`).
- Many control dependencies are Windows-oriented (`mcculw`, `nidaqmx`, `pypylon` in `setup.py`), matching the intended runtime platform.
- Calibration stack relies on NumPy/Pandas-heavy transforms plus optional visualization/tooling (`plotly`, `pyvista`, `nglview`, `scikit-learn`).

## High-Value Files To Read Before Editing
- `README.md` (scope, install modes, run/test commands)
- `pyccapt/control/CONTROL.md` and `pyccapt/calibration/CALIBRATION.md` (module boundaries)
- `pyccapt/control/core/runtime.py` and `pyccapt/control/core/share_variables.py` (control bootstrap/state model)
- `pyccapt/calibration/core/share_variables.py` (calibration shared-state contract)
- `tests/control/test_control_process_boundaries.py` and `tests/conftest.py` (process contracts and test selection behavior)

