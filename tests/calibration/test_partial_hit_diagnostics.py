"""Tests for partial_hit_diagnostics.tdc_pulse_completeness.

The function classifies raw-TDC pulses as complete (fired the full
delay-line channel set) or partial, grouping ticks by contiguous runs of
``start_counter`` so it is robust to the uint32 counter wrapping during
long acquisitions.
"""
import numpy as np
import pandas as pd

from pyccapt.calibration.data_tools.partial_hit_diagnostics import (
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
    assert s["complete"] == 1
    assert s["partial"] == 2
    assert s["with_dld_match"] == 1  # only the 4-channel pulse


def test_grouping_is_start_counter_wrap_safe():
    # Two complete pulses share start_counter=10 (a uint32 wrap) but are
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
    # One channel fired anywhere -> required set is {0}, so the pulse is complete.
    assert s["channels_required"] == 1
    assert s["complete"] == 1
    assert s["partial"] == 0
