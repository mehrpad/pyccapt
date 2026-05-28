"""Regression test for the partial-recovery start_counter-wrap fix.

Commit (IO4): ``merge_partial_tdc_into_dld`` previously grouped orphan
TDC ticks by *sorting* on ``start_counter``. Because start_counter is a
uint32 hardware counter that wraps every ~71 min, two physically
distinct pulses that happen to share a counter value (one before, one
after a wrap) were merged into a single "mega-pulse", producing
spurious paired hits. The fix groups by a run-length encode over a
CHANGE in start_counter in acquisition order, which is wrap-safe.

This test monkeypatches the per-pulse recovery kernel to return one
synthetic hit per pulse, so the assertion is purely about how many
distinct pulse groups the wrap-containing orphan set produces.
"""
import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.data_tools import partial_recovery as pr
from pyccapt.calibration.data_tools.data_loadcrop import (
    EVENT_GROUP_ID_COLUMN,
    TDC_HAS_DLD_MATCH_COLUMN,
)


class _FakeVariables:
    """Minimal stand-in for the shared-variables object."""

    def __init__(self, data, data_tdc):
        self.data = data
        self.data_tdc = data_tdc
        self.max_tof = 1000.0
        self.flight_path_length = 110.0

    def sync_from_data(self, *args, **kwargs):
        # The real method rebuilds derived arrays; for this test we only
        # care about variables.data after the merge, so make it a no-op.
        return self.data


def _make_orphan_tdc(start_counters):
    """Build a tdc frame of orphan ticks, 2 ticks per listed pulse.

    Ticks are laid out in ACQUISITION ORDER (the order of
    ``start_counters``); each pulse contributes two ticks on channels
    0 and 1 so _recover_for_pulse sees >= 2 ticks.
    """
    rows = []
    for sc in start_counters:
        for ch in (0, 1):
            rows.append(
                {
                    "channel": ch,
                    "time_data": 100.0 + ch,
                    "start_counter": np.uint32(sc),
                    "high_voltage (V)": 5000.0,
                    "pulse_v (V)": 0.0,
                    "pulse_l (pJ)": 0.0,
                    EVENT_GROUP_ID_COLUMN: -1,
                    TDC_HAS_DLD_MATCH_COLUMN: False,
                }
            )
    return pd.DataFrame(rows)


def test_partial_recovery_does_not_merge_pulses_across_counter_wrap(monkeypatch):
    # Three pulses in acquisition order. Pulse 0 and pulse 2 share
    # start_counter == 100; between them pulse 1 has counter 50 (the
    # counter wrapped). The OLD sort-by-counter logic would sort the two
    # sc=100 pulses adjacent and merge them into ONE group, yielding 2
    # recovered pulses. The wrap-safe logic must keep all THREE.
    start_counters = [100, 50, 100]

    tdc_df = _make_orphan_tdc(start_counters)
    # Minimal native DLD frame: empty but with the required gid column.
    dld_df = pd.DataFrame({
        "high_voltage (V)": pd.Series([], dtype=float),
        "pulse_v (V)": pd.Series([], dtype=float),
        "pulse_l (pJ)": pd.Series([], dtype=float),
        "t (ns)": pd.Series([], dtype=float),
        "x_det (cm)": pd.Series([], dtype=float),
        "y_det (cm)": pd.Series([], dtype=float),
        "start_counter": pd.Series([], dtype=int),
        "mc (Da)": pd.Series([], dtype=float),
        "mc_uc (Da)": pd.Series([], dtype=float),
        EVENT_GROUP_ID_COLUMN: pd.Series([], dtype=int),
    })

    variables = _FakeVariables(dld_df, tdc_df)

    # One synthetic recovered xy hit per pulse, so the number of distinct
    # GIDs in the recovered frame equals the number of pulse groups.
    def _fake_recover(pulse_channels, pulse_times, **kwargs):
        return [{
            "detector_axis": "xy",
            "x_det (cm)": 0.0,
            "y_det (cm)": 0.0,
            "tof (ns)": 100.0,
        }]

    monkeypatch.setattr(pr, "_recover_for_pulse", _fake_recover)
    # Avoid touching real detector-config files / mc conversion.
    monkeypatch.setattr(
        pr, "load_detector_constants", lambda *a, **k: {"detector_limit_cm": 4.0}
    )

    n_recovered = pr.merge_partial_tdc_into_dld(variables, verbose=False)

    # Three distinct pulses => three recovered rows => three unique GIDs.
    assert n_recovered == 3, (
        f"Expected 3 recovered pulses (wrap-safe grouping), got {n_recovered}; "
        f"the sc=100 pulses on either side of the wrap were likely merged."
    )
    recovered = variables.data
    gids = recovered[EVENT_GROUP_ID_COLUMN].to_numpy()
    assert len(np.unique(gids)) == 3, (
        f"Expected 3 unique event_group_id values, got {np.unique(gids)}."
    )
