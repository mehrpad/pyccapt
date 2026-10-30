"""Regression tests for the 2026-10 package-audit calibration fixes."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.core import calibration, calibration_advanced
from pyccapt.calibration.core.concentration_profile import (
    calculate_concentration_profile,
    calculate_roi_concentration_profile,
)
from pyccapt.calibration.core.guided_workflow import GuidedWorkflow
from pyccapt.calibration.core.share_variables import Variables
from pyccapt.calibration.data_tools import data_loadcrop


# --- multi-peak corrections must detect peaks at the requested bin size ----

def _close_peak_spectrum():
    """Four peaks, two of them 1 Da apart, with a known voltage drift."""
    rng = np.random.default_rng(7)
    labels = np.repeat([13.5, 27.0, 28.0, 56.0], 20000)
    voltage = rng.uniform(4000.0, 6000.0, labels.size)
    measured = labels * (1.0 + 0.002 * (voltage - 5000.0) / 1000.0) + rng.normal(0, 0.005, labels.size)
    return labels, voltage, measured


def test_multi_peak_voltage_correction_uses_bin_size_for_peak_detection():
    labels, voltage, measured = _close_peak_spectrum()
    variables = Variables()
    variables.dld_t_calib = measured.copy()
    variables.mc_calib = measured.copy()

    _, peaks = calibration.multi_peak_voltage_corr_main(
        voltage, variables, calibration_mode="mc", model="curve_fit",
        bin_size=0.01, n_peaks=4, prominence=10, distance=50,
    )

    # With the old hard-coded 0.1 Da detection bins, distance=50 bins meant
    # 5 Da spacing: 27/28 collapsed into one ~1.4 Da window.
    assert len(peaks) == 4
    assert sorted(round(p["position"]) for p in peaks) == [14, 27, 28, 56]
    assert all(p["x2"] - p["x1"] < 0.5 for p in peaks)


@pytest.mark.parametrize("function", ["multi_peak_voltage_corr_main", "multi_peak_bowl_corr_main"])
def test_multi_peak_corrections_forward_bin_size(monkeypatch, function):
    seen = {}

    def detect(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after detection")

    monkeypatch.setattr(calibration_advanced, "_auto_detect_peaks", detect)
    variables = SimpleNamespace(get_calibration_array=lambda mode: np.ones(10), bowl_sampling_mode="polar")
    args = (np.ones(10), variables) if function == "multi_peak_voltage_corr_main" else \
        (np.zeros(10), np.zeros(10), np.ones(10), variables, 8.0)
    with pytest.raises(RuntimeError, match="stop after detection"):
        getattr(calibration_advanced, function)(*args, bin_size=0.02)
    assert seen["hist_bin_size"] == 0.02


# --- concentration profiles must not count NaN masses as atoms -------------

def _ranges():
    return pd.DataFrame({"name": ["Al", "O"], "mc_low": [26.5, 15.5], "mc_up": [27.5, 16.5],
                         "element": [["Al"], ["O"]], "complex": [[1], [1]]})


def test_sequential_profile_reports_nan_mass_separately():
    mc = np.r_[np.full(50, 27.0), np.full(50, np.nan)]

    profile = calculate_concentration_profile(mc, _ranges(), ["element:Al", "unranged"], window_size=100)

    assert profile["Al (element)"].tolist() == [100.0]
    assert profile["Unranged"].tolist() == [0.0]
    assert profile["unranged_events"].tolist() == [0]
    assert profile["excluded_events"].tolist() == [50]
    assert profile["total_atom_equivalents"].tolist() == [50.0]
    assert profile.attrs["overall_percentages"]["Al (element)"] == 100.0
    assert profile.attrs["overall_excluded_events"] == 50


def test_finite_unranged_mass_still_counts_in_the_denominator():
    mc = np.array([27.0, 27.0, 99.0, np.nan])
    profile = calculate_concentration_profile(mc, _ranges(), ["element:Al"], window_size=4)
    assert np.isclose(profile["Al (element)"].iloc[0], 200 / 3)
    assert profile["unranged_events"].iloc[0] == 1
    assert profile["excluded_events"].iloc[0] == 1


def test_roi_profile_reports_nan_mass_separately():
    n = 40
    z = np.linspace(0.0, 9.9, n)
    mc = np.where(np.arange(n) % 2 == 0, 27.0, np.nan)

    profile = calculate_roi_concentration_profile(
        np.zeros(n), np.zeros(n), z, mc, _ranges(), ["element:Al"],
        transverse_center=(0.0, 0.0), transverse_size=(2.0, 2.0), bin_width=5.0,
    )

    assert profile["Al (element)"].tolist() == [100.0, 100.0]
    assert profile["excluded_events"].sum() == n // 2
    assert profile.attrs["overall_percentages"]["Al (element)"] == 100.0
    assert profile.attrs["roi_excluded_events"] == n // 2


def test_guided_workflow_does_not_call_missing_mass_unranged():
    stub = SimpleNamespace(
        active_mask=np.array([True, True, True, False]),
        accepted={"calibration": {"mass": np.array([27.0, np.nan, 99.0, np.nan])}},
    )
    counts = GuidedWorkflow._unassigned_counts(stub, np.array([0, -1, -1, -1]))
    assert counts == {"unranged_events": 1, "no_mass_events": 1}


# --- auto raw analysis: range-file windows are mass/charge -----------------

def test_range_file_windows_are_always_interpreted_as_mass_to_charge():
    helper = pytest.importorskip("pyccapt.calibration.tutorials.tutorials_helpers.helper_auto_raw_analysis")
    assert helper.effective_peak_units("range", "tof") == "mc"
    assert helper.effective_peak_units("range", "mc") == "mc"
    assert helper.effective_peak_units("manual", "tof") == "tof"
    assert helper.effective_peak_units("manual", "mc") == "mc"


# --- delta_p at a uint64 start-counter wrap ---------------------------------

def test_delta_p_handles_uint64_counter_wrap():
    # create_pandas_dataframe stores start_counter as uint64.
    data = pd.DataFrame({"start_counter": np.array([998, 999, 2, 2, 5], dtype=np.uint64)})

    delta_p, multi = data_loadcrop.calculate_ppi_and_ipp(data, max_start_counter=1000)

    assert delta_p.tolist() == [0, 1, 3, 0, 3]
    assert multi.tolist() == [1, 1, 2, 2, 1]
