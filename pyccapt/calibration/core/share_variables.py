"""Shared mutable state for the calibration workflow.

The notebook helpers and the calibration core pass a single ``Variables``
instance around instead of threading dozens of arrays through every
function. It holds, for the dataset currently loaded:

- the working arrays (``dld_t``, ``dld_high_voltage``, ``dld_x_det`` /
  ``dld_y_det``, ``mc``, ``mc_uc``, ``x``/``y``/``z``, ...),
- the per-mode calibrated arrays ``dld_t_calib`` (tof) and ``mc_calib``
  (m/c), plus ``*_backup`` snapshots used to revert a correction,
- the active peak range and per-mode calibration selection masks, and
- the ``initial_calibration_done_{mc,tof}`` flags that tell the
  auto-calibration buttons whether the global t0/flight-path step has
  already run on THIS dataset.

``sync_from_data`` rebuilds all of the above from a dataframe after a
load / crop / reset; it resets the per-dataset flags so a freshly loaded
dataset is not mistaken for an already-calibrated one. State/validation
problems raise the explicit calibration exceptions from ``exceptions``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from pyccapt.calibration.core.exceptions import CalibrationInputError, CalibrationStateError
from pyccapt.calibration.core.validation import CALIBRATION_MODES, ensure_choice
from pyccapt.calibration.path_utils import build_output_path, ensure_directory


def get_project_path() -> str:
    """Return the project root path by walking parents until `setup.py` is found."""
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "setup.py").is_file():
            return str(parent)
    return str(current.parent)


def _create_default_range_data() -> pd.DataFrame:
    """Create the default unranged entry with stable column dtypes."""
    range_data = pd.DataFrame(
        {
            "name": ["unranged0"],
            "ion": ["un"],
            "mass": [0.0],
            "mc": [0.0],
            "mc_low": [0.0],
            "mc_up": [400.0],
            "color": ["#000000"],
            "element": [["unranged"]],
            "complex": [[0]],
            "isotope": [[0]],
            "charge": [0],
        }
    )
    return range_data.astype(
        {
            "name": "str",
            "ion": "str",
            "mass": "float64",
            "mc": "float64",
            "mc_low": "float64",
            "mc_up": "float64",
            "color": "str",
            "element": "object",
            "complex": "object",
            "isotope": "object",
            "charge": "uint32",
        }
    )


class SharedVariablesBase:
    """Shared state helpers used by calibration and visualization workflows."""

    _CALIBRATION_ATTR = {
        "tof": "dld_t_calib",
        "mc": "mc_calib",
    }

    def get_calibration_array(self, calibration_mode: str) -> np.ndarray:
        """Return the selected calibrated array for a calibration mode."""
        mode = ensure_choice(calibration_mode, field_name="calibration_mode", allowed=CALIBRATION_MODES)
        values = getattr(self, self._CALIBRATION_ATTR[mode])
        values = np.asarray(values)
        if values.size == 0:
            raise CalibrationStateError(f"{self._CALIBRATION_ATTR[mode]!r} is empty")
        return values

    def position_capable_mask(self) -> np.ndarray:
        """Boolean mask of rows with both detector axes reconstructed.

        After ``merge_partial_tdc=True`` the DLD frame contains
        partial-recovered rows with ``NaN`` on ``x_det (cm)`` or
        ``y_det (cm)``. Any analysis that needs the detector position
        (bowl correction, adaptive spatial residual, 3-D reconstruction,
        FDM) MUST filter via this mask first; otherwise the NaN poisons
        polynomial fits, histogram-bin estimates, and clustering.

        Returns a boolean ndarray aligned with ``self.dld_x_det`` /
        ``self.dld_y_det``. When neither array is populated the result
        is an empty bool array.
        """
        x_det = np.asarray(getattr(self, "dld_x_det", []), dtype=float)
        y_det = np.asarray(getattr(self, "dld_y_det", []), dtype=float)
        if x_det.size == 0 or y_det.size == 0:
            return np.zeros(max(x_det.size, y_det.size), dtype=bool)
        if x_det.shape != y_det.shape:
            return np.zeros(min(x_det.size, y_det.size), dtype=bool)
        return np.isfinite(x_det) & np.isfinite(y_det)

    def position_capable_count(self) -> int:
        """Number of rows with both detector axes finite."""
        return int(self.position_capable_mask().sum())

    def set_peak_range(self, left: float, right: float) -> None:
        """Set selected peak window after validating numeric order."""
        try:
            left_value = float(left)
            right_value = float(right)
        except (TypeError, ValueError) as exc:
            raise CalibrationInputError("Peak range boundaries must be numeric") from exc
        if right_value <= left_value:
            raise CalibrationInputError(
                f"Invalid peak range: left={left_value}, right={right_value}. 'right' must be greater than 'left'."
            )
        self.selected_x1 = left_value
        self.selected_x2 = right_value

    def clear_peak_range(self) -> None:
        """Reset selected peak window to an unselected state."""
        self.selected_x1 = 0
        self.selected_x2 = 0

    @staticmethod
    def _coerce_peak_range(left: float, right: float) -> tuple[float, float]:
        """Validate and normalize numeric peak-range boundaries."""
        try:
            left_value = float(left)
            right_value = float(right)
        except (TypeError, ValueError) as exc:
            raise CalibrationInputError("Peak range boundaries must be numeric") from exc
        if right_value <= left_value:
            raise CalibrationInputError(
                f"Invalid peak range: left={left_value}, right={right_value}. 'right' must be greater than 'left'."
            )
        return left_value, right_value

    def set_calibration_peak_range(self, calibration_mode: str, left: float, right: float) -> tuple[float, float]:
        """Store a calibration-only copy of the active peak window for one mode."""
        mode = ensure_choice(calibration_mode, field_name="calibration_mode", allowed=CALIBRATION_MODES)
        left_value, right_value = self._coerce_peak_range(left, right)
        self.calibration_peak_ranges[mode] = (left_value, right_value)
        return left_value, right_value

    def get_calibration_peak_range(self, calibration_mode: str) -> tuple[float, float]:
        """Return the active peak window for calibration, preferring the stored snapshot."""
        mode = ensure_choice(calibration_mode, field_name="calibration_mode", allowed=CALIBRATION_MODES)
        stored = self.calibration_peak_ranges.get(mode)
        if stored is not None:
            return float(stored[0]), float(stored[1])
        self.ensure_valid_peak_range()
        return float(self.selected_x1), float(self.selected_x2)

    def clear_calibration_peak_range(self, calibration_mode: str | None = None) -> None:
        """Clear one or all stored calibration-only peak windows."""
        if calibration_mode is None:
            self.calibration_peak_ranges.clear()
            return
        mode = ensure_choice(calibration_mode, field_name="calibration_mode", allowed=CALIBRATION_MODES)
        self.calibration_peak_ranges.pop(mode, None)

    def ensure_valid_peak_range(self) -> None:
        """Validate that a peak range is selected and logically valid."""
        if self.selected_x1 == 0 and self.selected_x2 == 0:
            raise CalibrationStateError("No peak range selected")
        if self.selected_x2 <= self.selected_x1:
            raise CalibrationStateError(f"Invalid peak range: selected_x1={self.selected_x1}, selected_x2={self.selected_x2}")

    def build_calibration_mask(self, calibration_mode: str) -> np.ndarray:
        """Build a boolean mask from the selected peak range and calibration mode."""
        mode = ensure_choice(calibration_mode, field_name="calibration_mode", allowed=CALIBRATION_MODES)
        data = self.get_calibration_array(mode)
        override = self.calibration_selection_masks.get(mode)
        if override is not None:
            override = np.asarray(override, dtype=bool)
            if override.shape != data.shape:
                raise CalibrationStateError(
                    f"Locked calibration mask shape {override.shape} does not match data shape {data.shape}"
                )
            if not np.any(override):
                raise CalibrationStateError(f"Locked calibration mask is empty for calibration_mode={mode!r}")
            return override.copy()

        left, right = self.get_calibration_peak_range(mode)
        mask = np.logical_and(data > left, data < right)
        if not np.any(mask):
            raise CalibrationStateError(
                f"Selected peak range does not include any ions for calibration_mode={calibration_mode!r}"
            )
        return mask

    def set_calibration_selection_mask(self, calibration_mode: str, mask: np.ndarray) -> None:
        """Lock a boolean ion-selection mask for repeated calibration steps."""
        mode = ensure_choice(calibration_mode, field_name="calibration_mode", allowed=CALIBRATION_MODES)
        data = self.get_calibration_array(mode)
        locked_mask = np.asarray(mask, dtype=bool)
        if locked_mask.shape != data.shape:
            raise CalibrationInputError(f"Mask shape {locked_mask.shape} does not match calibration data shape {data.shape}")
        if not np.any(locked_mask):
            raise CalibrationInputError("Locked calibration mask cannot be empty")
        self.calibration_selection_masks[mode] = locked_mask.copy()

    def clear_calibration_selection_mask(self, calibration_mode: str | None = None) -> None:
        """Clear one or all locked calibration ion masks."""
        if calibration_mode is None:
            self.calibration_selection_masks.clear()
            return
        mode = ensure_choice(calibration_mode, field_name="calibration_mode", allowed=CALIBRATION_MODES)
        self.calibration_selection_masks.pop(mode, None)

    @staticmethod
    def _column_or_zeros(dataframe: pd.DataFrame, column: str, *, like: str | None = None) -> np.ndarray:
        """Return a dataframe column as ndarray, or zeros matched to another column length."""
        if column in dataframe.columns:
            return dataframe[column].to_numpy()
        if like is not None and like in dataframe.columns:
            return np.zeros(len(dataframe[like].to_numpy()))
        return np.zeros(len(dataframe))

    def sync_from_data(
        self, data: pd.DataFrame | None = None, *, update_backups: bool = False, clear_selection: bool = True
    ) -> pd.DataFrame:
        """Synchronize shared arrays from the current dataframe after crop/load/reset operations."""
        frame = self.data if data is None else data
        if frame is None:
            raise CalibrationStateError("No dataframe is available to synchronize shared variables")

        frame = frame.reset_index(drop=True).copy()
        self.data = frame
        if update_backups or self.data_backup is None:
            self.data_backup = frame.copy()

        self.dld_high_voltage = self._column_or_zeros(frame, "high_voltage (V)")
        if "pulse_v (V)" in frame.columns:
            self.dld_pulse_v = frame["pulse_v (V)"].to_numpy()
        elif "pulse" in frame.columns:
            self.dld_pulse_v = frame["pulse"].to_numpy()
        else:
            self.dld_pulse_v = np.zeros(len(frame))
        self.dld_pulse_l = self._column_or_zeros(frame, "pulse_l (pJ)", like="high_voltage (V)")
        self.dld_t = self._column_or_zeros(frame, "t (ns)")
        self.dld_t_c = self._column_or_zeros(frame, "t_c (ns)", like="t (ns)")
        self.dld_x_det = self._column_or_zeros(frame, "x_det (cm)", like="t (ns)")
        self.dld_y_det = self._column_or_zeros(frame, "y_det (cm)", like="t (ns)")

        self.mc = self._column_or_zeros(frame, "mc (Da)", like="t (ns)")
        if "mc_uc (Da)" in frame.columns:
            self.mc_uc = frame["mc_uc (Da)"].to_numpy()
        else:
            self.mc_uc = self.mc.copy()

        self.dld_t_calib = self.dld_t.copy()
        self.mc_calib = self.mc_uc.copy()
        if update_backups or self.dld_t_calib_backup.shape != self.dld_t_calib.shape:
            self.dld_t_calib_backup = self.dld_t_calib.copy()
        if update_backups or self.mc_calib_backup.shape != self.mc_calib.shape:
            self.mc_calib_backup = self.mc_calib.copy()

        self.x = self._column_or_zeros(frame, "x (nm)", like="t (ns)")
        self.y = self._column_or_zeros(frame, "y (nm)", like="t (ns)")
        self.z = self._column_or_zeros(frame, "z (nm)", like="t (ns)")

        self.mask = None
        self.AptHistPlotter = None
        self.peak_x = []
        self.peak_y = []
        self.peak_widths = []
        self.x_hist = None
        self.y_hist = None
        self.h_line_pos = []
        self.clear_calibration_selection_mask()
        self.clear_calibration_peak_range()
        # The "initial calibration has been run" flags are per-dataset; they
        # MUST be reset on dataset reload, otherwise the auto-calibration
        # helpers will skip initial calibration on the new dataset and apply
        # V/Bowl correction on top of uncalibrated t (ns).
        self.initial_calibration_done_mc = False
        self.initial_calibration_done_tof = False
        if clear_selection:
            self.clear_peak_range()
            self.selected_x_fdm = 0
            self.selected_y_fdm = 0
            self.roi_fdm = 0
        return frame

    def restore_data_from_backup(self) -> pd.DataFrame:
        """Restore the original dataframe snapshot and resynchronize shared arrays."""
        if self.data_backup is None:
            raise CalibrationStateError("No backup dataframe is available for restore")
        return self.sync_from_data(self.data_backup.copy(), update_backups=False, clear_selection=True)

    @staticmethod
    def _dir_as_string(directory: str | Path) -> str:
        """Return a normalized directory string with a trailing separator."""
        path = ensure_directory(directory)
        path_str = str(path)
        if not path_str.endswith((os.sep, "/", "\\")):
            path_str = path_str + os.sep
        return path_str

    def set_result_directory(self, directory: str | Path) -> str:
        """Set and return the normalized directory used for figures and plots."""
        self.result_path = self._dir_as_string(directory)
        return self.result_path

    def set_result_data_directory(self, directory: str | Path) -> str:
        """Set and return the normalized directory used for data exports."""
        self.result_data_path = self._dir_as_string(directory)
        return self.result_data_path

    def resolve_result_file(self, filename: str) -> str:
        """Resolve `filename` inside `result_path`."""
        base = self.result_path or self.last_directory
        return str(build_output_path(base, filename))

    def resolve_result_data_file(self, filename: str) -> str:
        """Resolve `filename` inside `result_data_path`."""
        base = self.result_data_path or self.last_directory
        return str(build_output_path(base, filename))


class Variables(SharedVariablesBase):
    """Container for shared variables across calibration workflows."""

    def __init__(self):
        self.pulse_mode = None

        self.selected_x_fdm = 0
        self.selected_y_fdm = 0
        self.roi_fdm = 0

        self.selected_x1 = 0
        self.selected_x2 = 0
        self.selected_y1 = 0
        self.selected_y2 = 0
        self.selected_z1 = 0
        self.selected_z2 = 0
        self.h_line_pos = []

        self.list_material = []
        self.charge = []
        self.element = []
        self.isotope = []

        self.peaks_x_selected = []
        self.peaks_index_list = []

        self.result_path = ""
        self.path = ""
        self.dataset_name = ""
        self.result_data_path = ""
        self.result_data_name = ""

        self.dld_t = np.zeros(0)
        self.dld_t_c = np.zeros(0)
        self.dld_x_det = np.zeros(0)
        self.dld_y_det = np.zeros(0)
        self.x = np.zeros(0)
        self.y = np.zeros(0)
        self.z = np.zeros(0)
        self.dld_high_voltage = np.zeros(0)
        self.dld_pulse_v = np.zeros(0)
        self.dld_pulse_l = np.zeros(0)
        self.dld_t_calib = np.zeros(0)
        self.dld_t_calib_backup = np.zeros(0)
        self.mc = np.zeros(0)
        self.mc_uc = np.zeros(0)
        self.mc_calib = np.zeros(0)
        self.mc_calib_backup = np.zeros(0)

        self.max_peak = 0
        self.max_tof = None
        self.peaks_index = 0
        self.peak_x = []
        self.peak_y = []
        self.peak_widths = []

        self.x_hist = None
        self.y_hist = None
        self.AptHistPlotter = None
        self.ions_list_data = None
        self.last_directory = get_project_path()
        # Paths chosen via the notebook Browse widgets (helper_data_loader
        # dataset_browser/range_browser). None until the user picks a file.
        self.dataset_path = None
        self.range_path = None

        self.plotly_3d_reconstruction = None
        self.data = None
        self.data_backup = None
        self.data_tdc = None
        self.data_tdc_backup = None
        self.max_mc = 400
        self.flight_path_length = None
        self.mask = None

        self.range_data = _create_default_range_data()
        self.range_data_backup = None
        self.animation_detector_html = None
        self.calibration_selection_masks = {}
        self.calibration_peak_ranges = {}
        self.bowl_sampling_mode = "cartesian"
        # Tracks whether the per-mode initial calibration has been run on the
        # current dataset, so the various auto-calibration buttons can
        # transparently do it first if needed (mc and tof have different
        # initial calibrations and are tracked independently).
        self.initial_calibration_done_mc = False
        self.initial_calibration_done_tof = False

    @property
    def dld_highVoltage(self):
        """Backward-compatible alias for legacy camelCase field names."""
        return self.dld_high_voltage

    @dld_highVoltage.setter
    def dld_highVoltage(self, value):
        self.dld_high_voltage = value

    @property
    def data_name(self):
        """Backward-compatible alias for `dataset_name`."""
        return self.dataset_name

    @data_name.setter
    def data_name(self, value):
        self.dataset_name = value
