import numpy as np
import pytest
from pathlib import Path

from pyccapt.calibration.core.exceptions import CalibrationInputError, CalibrationStateError
from pyccapt.calibration.core.share_variables import Variables


def test_set_peak_range_validates_order():
    variables = Variables()
    with pytest.raises(CalibrationInputError):
        variables.set_peak_range(20, 10)


def test_build_calibration_mask_for_tof_mode():
    variables = Variables()
    variables.dld_t_calib = np.array([10.0, 20.0, 30.0, 40.0])
    variables.set_peak_range(15, 35)

    mask = variables.build_calibration_mask("tof")

    assert mask.tolist() == [False, True, True, False]


def test_build_calibration_mask_raises_when_no_selected_ions():
    variables = Variables()
    variables.dld_t_calib = np.array([1.0, 2.0, 3.0])
    variables.set_peak_range(100, 120)

    with pytest.raises(CalibrationStateError):
        variables.build_calibration_mask("tof")


def test_default_range_data_contains_unranged_entry():
    variables = Variables()

    assert list(variables.range_data.columns) == [
        "name",
        "ion",
        "mass",
        "mc",
        "mc_low",
        "mc_up",
        "color",
        "element",
        "complex",
        "isotope",
        "charge",
    ]
    assert variables.range_data.iloc[0]["name"] == "unranged0"


def test_result_directory_helpers_are_cross_platform(tmp_path: Path):
    variables = Variables()
    figures_dir = tmp_path / "results" / "figures"

    normalized = variables.set_result_directory(figures_dir)
    resolved_file = variables.resolve_result_file("plot.png")

    assert figures_dir.exists()
    assert normalized.endswith(("/", "\\"))
    assert Path(resolved_file) == figures_dir / "plot.png"


def test_legacy_aliases_map_to_new_variable_names():
    variables = Variables()

    variables.dld_highVoltage = np.array([1.0, 2.0])
    variables.data_name = "dataset_01"

    assert variables.dld_high_voltage.tolist() == [1.0, 2.0]
    assert variables.dataset_name == "dataset_01"
