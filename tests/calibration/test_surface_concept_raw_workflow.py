import numpy as np
import pandas as pd

from pyccapt.calibration.data_tools._raw_workflow_surface_concept import (
    analyze_surface_concept_tdc_frame,
    plot_surface_concept_peak_ratio_table,
    summarize_surface_concept_peak_windows,
)
from pyccapt.calibration.data_tools._raw_workflow_common import compute_same_pulse_detector_separations
from pyccapt.calibration.data_tools.raw_data_surface_concept import (
    find_consecutive_sequences,
    find_nth_max_repeated_indices,
)


def test_find_consecutive_sequences_keeps_group_voltage_and_pulse():
    result = find_consecutive_sequences(
        start_counter=np.array([10, 10, 11], dtype=np.uint32),
        channel=np.array([0, 1, 0], dtype=np.uint32),
        time_data=np.array([100, 101, 200], dtype=np.uint32),
        high_voltage=np.array([1500.0, 1500.0, 2500.0], dtype=float),
        pulse=np.array([12.0, 12.0, 22.0], dtype=float),
        print_stats=False,
    )

    assert len(result) == 2
    assert result[0]["start_counter"] == [10, 10]
    assert result[0]["high_voltage"] == 1500.0
    assert result[0]["pulse"] == 12.0
    assert result[1]["high_voltage"] == 2500.0
    assert result[1]["pulse"] == 22.0


def test_analyze_surface_concept_tdc_frame_uses_loaded_pulse_v_column():
    tdc_df = pd.DataFrame(
        {
            "channel": np.array([0, 1, 2, 3], dtype=np.uint32),
            "start_counter": np.array([5, 5, 5, 5], dtype=np.uint32),
            "high_voltage (V)": np.array([2000.0, 2000.0, 2000.0, 2000.0], dtype=float),
            "pulse_v (V)": np.array([25.0, 25.0, 25.0, 25.0], dtype=float),
            "pulse_l (pJ)": np.zeros(4, dtype=float),
            "time_data": np.array([100, 110, 120, 130], dtype=np.uint32),
        }
    )

    analysis = analyze_surface_concept_tdc_frame(
        tdc_df,
        flight_path_length_mm=110.0,
        pulse_mode="voltage",
        t0=0.0,
        show_progress=False,
    )

    assert analysis["raw_summary"]["valid_four_channel_groups"] == 1
    assert len(analysis["hit_table"]) == 1
    hit = analysis["hit_table"].iloc[0]
    assert hit["dlts"] == 4
    assert hit["in_detector"]
    assert hit["pulse"] == 25.0
    assert np.isfinite(hit["tof (ns)"])
    assert np.isfinite(hit["mc (Da)"])


def test_find_nth_max_repeated_indices_is_one_based_for_rank():
    start, end, count, number = find_nth_max_repeated_indices([1, 1, 1, 2, 2, 3], 1)

    assert (start, end, count, number) == (0, 2, 3, 1)


def test_same_pulse_detector_separations_and_peak_summary():
    hit_table = pd.DataFrame(
        {
            "start_counter": [10, 10, 11, 12],
            "x_det (cm)": [0.0, 0.3, 0.5, 1.0],
            "y_det (cm)": [0.0, 0.4, 0.7, 1.2],
            "tof (ns)": [100.0, 101.0, 150.0, 200.0],
            "mc (Da)": [10.0, 10.2, 20.0, 30.0],
            "dlts": [4, 4, 2, 4],
            "in_detector": [True, True, True, False],
        }
    )
    diagnostics = pd.DataFrame(
        {
            "status": [
                "4 DLTS in detector",
                "4 DLTS in detector",
                "2 DLTS in detector",
                "4 DLTS outside detector",
                "unrecoverable",
            ],
            "dlts": [4, 4, 2, 4, 0],
        }
    )

    pair_table, summary = compute_same_pulse_detector_separations(
        hit_table,
        only_in_detector=True,
        dlts_values=[4],
        show_progress=False,
    )
    assert len(pair_table) == 1
    assert summary["pair_count"] == 1
    assert summary["min_dx"] == 0.3
    assert summary["min_dy"] == 0.4

    peak_summary = summarize_surface_concept_peak_windows(
        hit_table,
        diagnostics,
        windows=[{"label": "Peak 1", "min": 9.5, "max": 10.5}, {"label": "Peak 2", "min": 19.5, "max": 20.5}],
        signal_kind="mc",
        only_in_detector=True,
    )
    ratio_table = peak_summary["ratios"]
    assert list(ratio_table["Peak"]) == ["Peak 1", "Peak 2"]
    assert list(ratio_table["Four DLTS count"]) == [2, 0]
    assert list(ratio_table["Two DLTS count"]) == [0, 1]
    assert peak_summary["outside_detector_count"] == 1
    assert peak_summary["unrecoverable_count"] == 1

    figure = plot_surface_concept_peak_ratio_table(ratio_table)
    assert figure is not None
