"""Tests for Surface Concept 3-of-4 (time-sum) partial-hit recovery.

A 2-D crossed delay-line detector reconstructs a full (x, y) position from
only three of the four channel timestamps: the two delay lines share the
same total propagation time (t0 + t1 = t2 + t3), so a missing delay-line
end is recovered as ``sum_complete_axis - t_present``. These tests pin the
single-pulse recovery, the multi-hit case (a full ion plus a 3-channel ion
in one pulse), the opt-out, the detector gate, and the row labelling.
"""
import numpy as np
import pytest

from pyccapt.calibration.data_tools import _raw_workflow_surface_concept as sc
from pyccapt.calibration.data_tools import partial_recovery


def _pos(a, b):
    return sc._surface_concept_position_from_pair(a, b)


def test_single_three_channel_pulse_recovers_full_xy():
    # X complete (ch0, ch1) + orphan ch2 -> recover ch3 via time-sum.
    ch = np.array([0, 1, 2], dtype=np.int64)
    t = np.array([100, 140, 130], dtype=np.int64)
    hits = sc._recover_surface_concept_partial_hits(ch, t, detector_radius_cm=100.0)

    assert len(hits) == 1
    hit = hits[0]
    assert hit["detector_axis"] == "xy"
    assert hit["recovery_method"] == "3of4"
    t3 = (100 + 140) - 130  # recovered missing end
    assert abs(hit["x_det (cm)"] - _pos(100, 140)) < 1e-9
    assert abs(hit["y_det (cm)"] - _pos(130, t3)) < 1e-9


def test_multihit_full_plus_three_channel():
    # One pulse, 7 ticks: full ion {0,1,2,3} (axis sums match -> combines)
    # plus a 3-channel ion {1,2,3} -> recover its ch0.
    ch = np.array([0, 1, 2, 3, 1, 2, 3], dtype=np.int64)
    t = np.array([100, 140, 130, 110, 200, 230, 170], dtype=np.int64)
    hits = sc._recover_surface_concept_partial_hits(ch, t, detector_radius_cm=100.0)

    assert len(hits) == 2
    assert sorted(h["detector_axis"] for h in hits) == ["xy", "xy"]
    methods = sorted(str(h.get("recovery_method")) for h in hits)
    assert methods == ["3of4", "None"]  # one normal xy + one 3-of-4
    recovered = next(h for h in hits if h.get("recovery_method") == "3of4")
    # recovered ch0 = (230+170) - 200 = 200 -> x = pos(200, 200) = 0 (centre)
    assert abs(recovered["x_det (cm)"] - _pos(200, 200)) < 1e-9


def test_opt_out_keeps_one_dimensional_partial():
    ch = np.array([0, 1, 2], dtype=np.int64)
    t = np.array([100, 140, 130], dtype=np.int64)
    hits = sc._recover_surface_concept_partial_hits(
        ch, t, detector_radius_cm=100.0, recover_three_channel=False
    )
    # Without 3-of-4 recovery the complete X axis is just a 1-D x partial.
    assert len(hits) == 1
    assert hits[0]["detector_axis"] == "x"
    assert np.isnan(hits[0]["y_det (cm)"])


def test_detector_gate_rejects_out_of_area_recovery():
    # Tight radius: the recovered y position falls outside the detector,
    # so no full xy hit is produced (it stays a 1-D partial instead).
    ch = np.array([0, 1, 2], dtype=np.int64)
    t = np.array([100, 140, 130], dtype=np.int64)
    hits = sc._recover_surface_concept_partial_hits(ch, t, detector_radius_cm=0.001)
    assert all(h.get("recovery_method") != "3of4" for h in hits)


def test_diagnostics_path_does_not_three_channel_recover():
    # combine_axes=False (diagnostics/reporting) must not promote to xy.
    ch = np.array([0, 1, 2], dtype=np.int64)
    t = np.array([100, 140, 130], dtype=np.int64)
    hits = sc._recover_surface_concept_partial_hits(
        ch, t, detector_radius_cm=float("inf"), combine_axes=False
    )
    assert all(h["detector_axis"] != "xy" for h in hits)


def test_build_recovered_row_labels_3of4():
    hit = {
        "x_det (cm)": 0.5,
        "y_det (cm)": -0.5,
        "tof (ns)": 100.0,
        "detector_axis": "xy",
        "recovery_method": "3of4",
    }
    columns = ["high_voltage (V)", "pulse_v (V)", "pulse_l (pJ)", "t (ns)",
               "x_det (cm)", "y_det (cm)", "start_counter", "mc (Da)", "mc_uc (Da)",
               partial_recovery.EVENT_GROUP_ID_COLUMN,
               partial_recovery.DLTS_COLUMN, partial_recovery.DLTS_QUALITY_COLUMN]
    row = partial_recovery._build_recovered_row(
        start_counter=1, high_voltage=5000.0, pulse_v=0.0, pulse_l=0.0,
        recovered_hit=hit, new_gid=7, dld_columns=columns,
    )
    assert int(row[partial_recovery.DLTS_COLUMN]) == 4
    assert row[partial_recovery.DLTS_QUALITY_COLUMN] == "recovered_xy_3of4"
