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

    def set_peak_range(self, left: float, right: float) -> None:
        """Set selected peak window after validating numeric order."""
        try:
            left_value = float(left)
            right_value = float(right)
        except (TypeError, ValueError) as exc:
            raise CalibrationInputError("Peak range boundaries must be numeric") from exc
        if right_value <= left_value:
            raise CalibrationInputError(
                f"Invalid peak range: left={left_value}, right={right_value}. "
                "'right' must be greater than 'left'."
            )
        self.selected_x1 = left_value
        self.selected_x2 = right_value

    def clear_peak_range(self) -> None:
        """Reset selected peak window to an unselected state."""
        self.selected_x1 = 0
        self.selected_x2 = 0

    def ensure_valid_peak_range(self) -> None:
        """Validate that a peak range is selected and logically valid."""
        if self.selected_x1 == 0 and self.selected_x2 == 0:
            raise CalibrationStateError("No peak range selected")
        if self.selected_x2 <= self.selected_x1:
            raise CalibrationStateError(
                f"Invalid peak range: selected_x1={self.selected_x1}, selected_x2={self.selected_x2}"
            )

    def build_calibration_mask(self, calibration_mode: str) -> np.ndarray:
        """Build a boolean mask from the selected peak range and calibration mode."""
        self.ensure_valid_peak_range()
        data = self.get_calibration_array(calibration_mode)
        mask = np.logical_and(data > self.selected_x1, data < self.selected_x2)
        if not np.any(mask):
            raise CalibrationStateError(
                "Selected peak range does not include any ions for "
                f"calibration_mode={calibration_mode!r}"
            )
        return mask

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

        self.plotly_3d_reconstruction = None
        self.data = None
        self.data_backup = None
        self.max_mc = 400
        self.flight_path_length = None
        self.mask = None

        self.range_data = _create_default_range_data()
        self.range_data_backup = None
        self.animation_detector_html = None

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

