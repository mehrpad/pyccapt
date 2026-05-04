import math

import numpy as np
import pandas as pd

from pyccapt.calibration.core import flight_path_t0


class DummyVariables:
    def __init__(self):
        self.peak_widths = (
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
            np.array([3.0]),
        )
        self.peak_x = np.array([12.0])
        self.x_hist = np.array([0.0, 10.0, 11.0, 13.0, 20.0])
        self.peaks_index_list = [0]
        self.mc_uc = np.array([9.5, 10.5, 11.5, 12.4, 18.0])


def test_build_selected_peak_ranges_uses_histogram_windows():
    variables = DummyVariables()
    selected = flight_path_t0.build_selected_peak_ranges(variables)

    assert len(selected) == 1
    assert math.isclose(selected[0]["left"], 10.0)
    assert math.isclose(selected[0]["right"], 13.0)
    assert selected[0]["num_ions"] == 3


def test_fit_flight_path_and_t0_recovers_linear_solution():
    peak_table = pd.DataFrame(
        {
            "ideal_mc": [27.0, 27.0, 56.0, 56.0],
            "high_voltage_v": [6000.0, 6500.0, 7000.0, 7500.0],
            "pulse_v": [0.0, 0.0, 0.0, 0.0],
        }
    )
    factor = flight_path_t0.compute_time_scaling_factor(
        peak_table["ideal_mc"],
        peak_table["high_voltage_v"],
        peak_table["pulse_v"],
        "laser",
    )
    expected_t0 = 32.5
    expected_flight_path = 109.4
    peak_table["tof_ns"] = expected_t0 + expected_flight_path * factor

    fixed = flight_path_t0.estimate_fixed_path_t0(
        peak_table,
        flight_path_length_mm=expected_flight_path,
        pulse_mode="laser",
    )
    fitted = flight_path_t0.fit_flight_path_and_t0(peak_table, pulse_mode="laser")

    assert abs(fixed["t0_ns"] - expected_t0) < 1e-9
    assert abs(fitted["t0_ns"] - expected_t0) < 1e-9
    assert abs(fitted["flight_path_length_mm"] - expected_flight_path) < 1e-9
    assert fitted["r_squared"] > 0.999999


def test_filter_peak_regression_table_keeps_detector_center_hits():
    peak_table = pd.DataFrame(
        {
            "peak_label": ["A", "A", "B"],
            "peak_index": [0, 0, 1],
            "measured_mc": [12.0, 12.0, 24.0],
            "ideal_mc": [12.0, 12.0, 24.0],
            "window_left": [11.0, 11.0, 23.0],
            "window_right": [13.0, 13.0, 25.0],
            "tof_ns": [100.0, 101.0, 200.0],
            "x_det_cm": [0.1, 0.8, 0.2],
            "y_det_cm": [0.1, 0.7, 0.2],
            "high_voltage_v": [6000.0, 6000.0, 6100.0],
            "pulse_v": [0.0, 0.0, 0.0],
        }
    )

    filtered = flight_path_t0.filter_peak_regression_table(
        peak_table,
        center_radius_cm=0.5,
        voltage_tolerance_v=60.0,
    )

    assert len(filtered) == 2
    assert filtered["detector_radius_cm"].max() <= 0.5
