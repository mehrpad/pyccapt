"""Tests for partial_hit_diagnostics.tdc_pulse_completeness.

The function classifies raw-TDC pulses as complete (fired the full
delay-line channel set) or partial, grouping ticks by contiguous runs of
``start_counter`` so it is robust to the uint32 counter wrapping during
long acquisitions.
"""
import numpy as np
import pandas as pd

from pyccapt.calibration.data_tools._raw_workflow_surface_concept import (
    _surface_concept_hit_from_time_data as _hit4,
)
from pyccapt.calibration.data_tools.partial_hit_diagnostics import (
    matched_dld_tdc_residuals,
    tdc_pulse_completeness,
)


def _tdc_frame(pulses):
    """Build a tdc frame from [(start_counter, [channels]), ...] in order."""
    rows = []
    for sc, chans in pulses:
        for ch in chans:
            rows.append(
                {
                    "channel": ch,
                    "start_counter": sc,
                    "time_data": 100 + ch,  # non-zero -> fired
                    "has_dld_match": len(chans) == 4,
                }
            )
    return pd.DataFrame(rows)


def test_complete_vs_partial_basic():
    tdc = _tdc_frame([(1, [0, 1, 2, 3]), (2, [0, 1]), (3, [2, 3])])
    s = tdc_pulse_completeness(tdc)
    assert s["channels_required"] == 4
    assert s["total_pulses"] == 3
    assert s["complete"] == 1            # the 4-channel pulse fired all channels
    assert s["partial"] == 2             # the two 2-channel pulses
    assert s["with_dld_match"] == 1      # only the 4-channel pulse
    assert s["channel_histogram"] == {4: 1, 2: 2}


def test_channel_histogram_reports_every_count():
    # A 3-channel pulse fires fewer than all 4 channels, so by the raw-stream
    # completeness criterion it is 'partial'; the histogram still records it.
    tdc = _tdc_frame([(1, [0, 1, 2, 3]), (2, [0, 1, 2])])
    s = tdc_pulse_completeness(tdc)
    assert s["channels_required"] == 4
    assert s["complete"] == 1            # only the 4-channel pulse
    assert s["partial"] == 1             # the 3-channel pulse
    assert s["channel_histogram"] == {4: 1, 3: 1}


def test_grouping_is_start_counter_wrap_safe():
    # Two complete pulses share start_counter=10 (a counter wrap) but are
    # separated in acquisition order by a different pulse. Contiguous-run
    # grouping must keep them as two distinct pulses, not merge them.
    tdc = _tdc_frame([(10, [0, 1, 2, 3]), (11, [0, 1]), (10, [0, 1, 2, 3])])
    s = tdc_pulse_completeness(tdc)
    assert s["total_pulses"] == 3
    assert s["complete"] == 2
    assert s["partial"] == 1


def test_roentdek_style_zero_time_data_counts_as_not_fired():
    # RoentDek flat layout writes a row for every channel; non-fired
    # channels have time_data == 0 and must not count toward completeness.
    rows = []
    # Pulse 1: channels 0-3 all fired.
    for ch in range(4):
        rows.append({"channel": ch, "start_counter": 1, "time_data": 10 + ch})
    # Pulse 2: rows for all 4 channels present, but only 0,1 actually fired.
    for ch in range(4):
        rows.append(
            {"channel": ch, "start_counter": 2, "time_data": (10 + ch) if ch < 2 else 0}
        )
    tdc = pd.DataFrame(rows)
    s = tdc_pulse_completeness(tdc)
    assert s["channels_required"] == 4
    assert s["total_pulses"] == 2
    assert s["complete"] == 1
    assert s["partial"] == 1


def test_empty_and_missing_inputs():
    assert tdc_pulse_completeness(None)["total_pulses"] == 0
    assert tdc_pulse_completeness(pd.DataFrame())["total_pulses"] == 0
    # Missing required columns -> zeros, no crash.
    assert tdc_pulse_completeness(pd.DataFrame({"x": [1, 2]}))["total_pulses"] == 0


def test_single_row_frame():
    tdc = _tdc_frame([(1, [0])])
    s = tdc_pulse_completeness(tdc)
    assert s["total_pulses"] == 1
    # Only channel 0 fires anywhere -> required set is {0}; the single pulse
    # fired all required channels, so it is complete.
    assert s["channels_required"] == 1
    assert s["complete"] == 1
    assert s["partial"] == 0


# ---------------------------------------------------------------------------
# matched_dld_tdc_residuals — TDC-vs-DLD match-quality cross-check
# ---------------------------------------------------------------------------


def _make_frames(pulses, dld_events):
    """Build (dld_df, tdc_df) from compact pulse / event specs.

    ``pulses``     : list of dict(sc, gid, channels, times, has_match).
    ``dld_events`` : list of dict(gid, x, y, t, [quality]).
    """
    tdc_rows = []
    for p in pulses:
        for ch, tm in zip(p["channels"], p["times"]):
            tdc_rows.append(
                {
                    "channel": int(ch),
                    "start_counter": int(p["sc"]),
                    "time_data": int(tm),
                    "event_group_id": int(p["gid"]),
                    "has_dld_match": bool(p["has_match"]),
                }
            )
    tdc = pd.DataFrame(tdc_rows)

    dld_rows = []
    for e in dld_events:
        row = {
            "x_det (cm)": float(e["x"]),
            "y_det (cm)": float(e["y"]),
            "t (ns)": float(e["t"]),
            "event_group_id": int(e["gid"]),
        }
        if "quality" in e:
            row["dlts_quality"] = e["quality"]
        dld_rows.append(row)
    dld = pd.DataFrame(dld_rows)
    return dld, tdc


def test_residuals_zero_for_correctly_linked_pulse():
    # A clean 4-channel matched pulse whose DLD event holds exactly the values
    # the same stops reconstruct -> residuals are (numerically) zero. The
    # stops are listed OUT of channel order to exercise the per-pulse sort.
    x, y, t = _hit4([100, 140, 130, 110])  # times for ch0, ch1, ch2, ch3
    pulses = [
        dict(sc=5, gid=0, channels=[2, 0, 3, 1], times=[130, 100, 110, 140], has_match=True),
    ]
    dld_events = [dict(gid=0, x=x, y=y, t=t, quality="native")]
    dld, tdc = _make_frames(pulses, dld_events)

    res = matched_dld_tdc_residuals(dld, tdc)
    assert res["n_compared"] == 1
    assert res["dx_max_cm"] < 1e-9
    assert res["dy_max_cm"] < 1e-9
    assert res["dtof_max_ns"] < 1e-9
    assert res["frac_x_within_tol"] == 1.0
    assert res["frac_y_within_tol"] == 1.0
    assert res["frac_tof_within_tol"] == 1.0


def test_multihit_pulse_is_excluded():
    # One clean 4-stop pulse plus an 8-stop multi-hit pulse. Only the clean
    # pulse is an unambiguous ground truth, so only it is cross-checked.
    x0, y0, t0 = _hit4([100, 140, 130, 110])
    pulses = [
        dict(sc=5, gid=0, channels=[0, 1, 2, 3], times=[100, 140, 130, 110], has_match=True),
        dict(
            sc=6, gid=1,
            channels=[0, 1, 2, 3, 0, 1, 2, 3],
            times=[200, 260, 250, 210, 300, 360, 350, 310],
            has_match=True,
        ),
    ]
    dld_events = [
        dict(gid=0, x=x0, y=y0, t=t0, quality="native"),
        dict(gid=1, x=0.0, y=0.0, t=1500.0, quality="native"),
    ]
    dld, tdc = _make_frames(pulses, dld_events)
    res = matched_dld_tdc_residuals(dld, tdc)
    assert res["n_compared"] == 1


def test_mislinked_pulse_shows_large_residual():
    # The stops are fine but the DLD event records a position/ToF that does
    # NOT correspond to them -> the residual is large and nothing is within
    # tolerance. This is the signature of an incorrect match.
    pulses = [
        dict(sc=5, gid=0, channels=[0, 1, 2, 3], times=[100, 140, 130, 110], has_match=True),
    ]
    dld_events = [dict(gid=0, x=2.0, y=-2.0, t=999.0, quality="native")]
    dld, tdc = _make_frames(pulses, dld_events)
    res = matched_dld_tdc_residuals(dld, tdc)
    assert res["n_compared"] == 1
    assert res["dx_max_cm"] > 0.01
    assert res["dtof_max_ns"] > 1.0
    assert res["frac_tof_within_tol"] == 0.0


def test_recovered_rows_excluded_unless_requested():
    # gid 1's DLD row is a recovered hit (its position came from these stops),
    # so comparing it to the stops is circular -> excluded by default; counted
    # only when only_native=False.
    x0, y0, t0 = _hit4([100, 140, 130, 110])
    x1, y1, t1 = _hit4([200, 260, 250, 210])
    pulses = [
        dict(sc=5, gid=0, channels=[0, 1, 2, 3], times=[100, 140, 130, 110], has_match=True),
        dict(sc=6, gid=1, channels=[0, 1, 2, 3], times=[200, 260, 250, 210], has_match=True),
    ]
    dld_events = [
        dict(gid=0, x=x0, y=y0, t=t0, quality="native"),
        dict(gid=1, x=x1, y=y1, t=t1, quality="recovered_xy"),
    ]
    dld, tdc = _make_frames(pulses, dld_events)
    assert matched_dld_tdc_residuals(dld, tdc)["n_compared"] == 1
    assert matched_dld_tdc_residuals(dld, tdc, only_native=False)["n_compared"] == 2


def test_unmatched_and_ambiguous_pulses_excluded():
    # has_match=False -> not a matched pulse; and a gid that owns two DLD rows
    # is ambiguous. Neither is cross-checked.
    x0, y0, t0 = _hit4([100, 140, 130, 110])
    # Unmatched pulse.
    pulses_unmatched = [
        dict(sc=5, gid=0, channels=[0, 1, 2, 3], times=[100, 140, 130, 110], has_match=False),
    ]
    dld_unmatched = [dict(gid=0, x=x0, y=y0, t=t0, quality="native")]
    dld_u, tdc_u = _make_frames(pulses_unmatched, dld_unmatched)
    assert matched_dld_tdc_residuals(dld_u, tdc_u)["n_compared"] == 0

    # gid owns two DLD rows -> ambiguous mapping.
    pulses_ambig = [
        dict(sc=5, gid=0, channels=[0, 1, 2, 3], times=[100, 140, 130, 110], has_match=True),
    ]
    dld_ambig = [
        dict(gid=0, x=x0, y=y0, t=t0, quality="native"),
        dict(gid=0, x=x0, y=y0, t=t0, quality="native"),
    ]
    dld_a, tdc_a = _make_frames(pulses_ambig, dld_ambig)
    assert matched_dld_tdc_residuals(dld_a, tdc_a)["n_compared"] == 0


def test_missing_inputs_report_reason():
    res = matched_dld_tdc_residuals(pd.DataFrame(), None)
    assert res["n_compared"] == 0
    assert res["reason"]
    res2 = matched_dld_tdc_residuals(None, pd.DataFrame({"x": [1]}))
    assert res2["n_compared"] == 0
    assert res2["reason"]
