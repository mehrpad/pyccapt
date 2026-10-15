"""Reproducible diagnostic boundaries: explicit labels and no accepted mutations."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[2] / "tools/validate_calibration_comparison.py"
    spec = importlib.util.spec_from_file_location("calibration_comparison_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def catalogue_and_ranges():
    catalogue = pd.DataFrame(dict(element=["Al", "Fe"], isotope=[27, 56], weight=[26.98, 55.94]))
    ranges = pd.DataFrame(dict(element=[["Al"], ["Fe"]], isotope=[[27], [56]], complex=[[1], [1]],
                               charge=[1, 1], ion=["Al27+", "Fe56+"], mc_low=[26., 55.], mc_up=[28., 57.]))
    return catalogue, ranges


def test_reference_selection_preserves_half_open_windows_and_catalogue_masses(runner):
    catalogue, ranges = catalogue_and_ranges()
    frame = pd.DataFrame({"mc (Da)": [25., 26., 27., 28., 55., 57.]})
    masks, details = runner.reference_masks(frame, ranges, [0, 1], catalogue)
    np.testing.assert_array_equal(masks[26.98], [False, True, True, False, False, False])
    np.testing.assert_array_equal(masks[55.94], [False, False, False, False, True, False])
    assert [row["events"] for row in details] == [2, 1]


@pytest.mark.parametrize("problem", ["duplicate_row", "out_of_bounds", "overlap", "empty", "missing_isotope", "charge", "composition"])
def test_invalid_assumed_references_fail_before_fitting(runner, problem):
    catalogue, ranges = catalogue_and_ranges()
    frame = pd.DataFrame({"mc (Da)": [26.98, 55.94]})
    rows = [0, 1]
    if problem == "duplicate_row":
        rows = [0, 0]
    elif problem == "out_of_bounds":
        rows = [-1, 1]
    elif problem == "overlap":
        ranges.loc[1, "mc_low"] = 26.
    elif problem == "empty":
        frame = frame.iloc[:1]
    elif problem == "missing_isotope":
        catalogue = catalogue.iloc[:1]
    elif problem == "charge":
        ranges.loc[1, "charge"] = 0
    else:
        ranges.at[0, "complex"] = []
    with pytest.raises(ValueError):
        runner.reference_masks(frame, ranges, rows, catalogue)


def test_output_collision_does_not_read_or_overwrite_input(runner, tmp_path):
    source = tmp_path / "source.h5"
    source.write_bytes(b"original")
    with pytest.raises(ValueError, match="overwrite"):
        runner.compare(source, tmp_path / "ranges.h5", source, rows=[0, 1], flight_mm=110.,
                       t0_ns=38., pulse_mode="voltage")
    assert source.read_bytes() == b"original"


def test_hardlink_output_collision_preserves_input(runner, tmp_path):
    source, alias = tmp_path / "source.h5", tmp_path / "alias.json"
    source.write_bytes(b"original")
    alias.hardlink_to(source)
    with pytest.raises(ValueError, match="overwrite"):
        runner.compare(source, tmp_path / "ranges.h5", alias, rows=[0, 1], flight_mm=110.,
                       t0_ns=38., pulse_mode="voltage")
    assert source.read_bytes() == alias.read_bytes() == b"original"


def test_conditional_diagnostic_holdouts_never_accept_or_apply(runner, monkeypatch, tmp_path):
    from pyccapt.calibration.core import flight_path_t0
    from pyccapt.calibration.core.guided_calibration import GuidedCalibrationSession
    from pyccapt.calibration.core.multireference_tof import physical_tof_baseline

    catalogue, ranges = catalogue_and_ranges()
    monkeypatch.setattr(flight_path_t0, "load_isotope_table", lambda: catalogue)
    def forbidden(*args, **kwargs):
        raise AssertionError("A diagnostic must not accept or apply the conditional calibration")
    monkeypatch.setattr(GuidedCalibrationSession, "accept", forbidden)
    monkeypatch.setattr(GuidedCalibrationSession, "apply", forbidden)
    rng = np.random.default_rng(24)
    n = 1800
    mass = np.tile([26.98, 55.94], n//2)
    voltage = rng.uniform(4000, 6000, n)
    x, y = rng.uniform(-2., 2., (2, n))
    pulse = voltage * .2
    baseline = physical_tof_baseline(voltage, x, y, 110., pulse, "voltage")
    frame = pd.DataFrame({"mc (Da)": mass*1.001, "mc_uc (Da)": mass,
                          "t (ns)": 38.+baseline*np.sqrt(mass), "high_voltage (V)": voltage,
                          "pulse_v (V)": pulse, "x_det (cm)": x, "y_det (cm)": y})
    source, ranges_path = tmp_path / "source.h5", tmp_path / "ranges.h5"
    frame.to_hdf(source, key="df")
    with pytest.warns(pd.errors.PerformanceWarning):
        ranges.to_hdf(ranges_path, key="df")
    destination = tmp_path / "report.json"
    result = runner.compare(source, ranges_path, destination, rows=[0, 1], flight_mm=110.,
                            t0_ns=38., pulse_mode="voltage")
    assert result["state"] == "confirmed"
    assert result["source_files_unchanged"] and result["active_calibration_unchanged"]
    assert result["grouping_benchmark"]["exact_values_and_order"]
    assert result["physical_vs_saved_uncorrected"]["allclose_rtol_1e_12"]
    assert result["confirmation"]["passed"]
    assert json.loads(destination.read_text())["assumptions"] == result["assumptions"]
    for row in result["selection_comparison"]:
        assert row["candidate"]["relative_rmse"] < 1e-8
