import numpy as np
import pandas as pd

from pyccapt.calibration.data_tools._raw_workflow_roentdek import (
    analyze_roentdek_tdc_frame,
    roentdek_processed_to_hit_table,
)


def test_analyze_roentdek_tdc_frame_normalizes_zero_based_channels():
    tdc_df = pd.DataFrame(
        {
            "channel": np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32),
            "start_counter": np.array([7, 7, 7, 7, 7, 7], dtype=np.uint32),
            "high_voltage (V)": np.full(6, 2000.0, dtype=float),
            "pulse_v (V)": np.full(6, 20.0, dtype=float),
        }
    )

    analysis = analyze_roentdek_tdc_frame(tdc_df, show_progress=False)

    assert analysis["raw_summary"]["matched_pattern_events"] == 1
    assert analysis["counters"]["dld6"][6] == 1
    assert analysis["events"][0]["dlts"] == [6]


def test_roentdek_processed_to_hit_table_maps_event_group_dlts():
    dld_df = pd.DataFrame(
        {
            "event_group_id": [0, 1],
            "start_counter": [10, 11],
            "t (ns)": [100.0, 120.0],
            "mc (Da)": [20.0, 30.0],
            "x_det (cm)": [0.1, 0.2],
            "y_det (cm)": [0.3, 0.4],
            "high_voltage (V)": [1000.0, 1100.0],
            "pulse_v (V)": [5.0, 6.0],
        }
    )
    analysis = {
        "events": [
            {"event_group_id": 0, "dlts": [6]},
            {"event_group_id": 1, "dlts": [4]},
        ]
    }

    hit_table = roentdek_processed_to_hit_table(dld_df, analysis)

    assert list(hit_table["dlts"]) == [6, 4]
    assert list(hit_table["tof (ns)"]) == [100.0, 120.0]
    assert list(hit_table["mc (Da)"]) == [20.0, 30.0]
