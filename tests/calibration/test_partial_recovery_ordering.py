"""Tests that partial-hit recovery preserves acquisition order.

``start_counter`` is a periodic counter that resets many times during a
run (so it is NOT a global timestamp). The merge must insert recovered
atoms at their true acquisition position -- right after the native rows
of the most recent matched pulse -- and must NEVER sort the merged frame
by ``start_counter`` (which would scramble the event sequence and destroy
the reconstructed z/depth). It must also, by default, merge only full
(x, y) hits and skip 1-D partials that have a NaN detector coordinate.
"""
import numpy as np
import pandas as pd

from pyccapt.calibration.data_tools import partial_recovery as pr

EGID = pr.EVENT_GROUP_ID_COLUMN
HASM = pr.TDC_HAS_DLD_MATCH_COLUMN


class _FakeVariables:
    def __init__(self, data, data_tdc):
        self.data = data
        self.data_tdc = data_tdc
        self.data_backup = None
        self.data_tdc_backup = None
        self.max_tof = 100000.0
        self.flight_path_length = 110.0

    def sync_from_data(self, *args, **kwargs):
        return self.data


def _native_dld(gids, start_counters):
    n = len(gids)
    return pd.DataFrame({
        "x (nm)": np.zeros(n),
        "y (nm)": np.zeros(n),
        "z (nm)": np.zeros(n),
        "mc (Da)": np.zeros(n),
        "mc_uc (Da)": np.zeros(n),
        "high_voltage (V)": np.full(n, 5000.0),
        "pulse_v (V)": np.zeros(n),
        "pulse_l (pJ)": np.zeros(n),
        "t (ns)": np.full(n, 1000.0),
        "x_det (cm)": np.full(n, 0.1),
        "y_det (cm)": np.full(n, 0.1),
        "start_counter": np.asarray(start_counters, dtype=np.int64),
        EGID: np.asarray(gids, dtype=np.int64),
        # tag native rows so they sort first via x_det marker below
        "native_marker": np.arange(n),
    })


def _tdc_rows(pulses):
    """pulses: list of (start_counter, gid, has_match, [channels])."""
    rows = []
    for sc, gid, has_match, chans in pulses:
        for ch in chans:
            rows.append({
                "channel": ch,
                "start_counter": sc,
                "time_data": 100 + ch * 10,
                "high_voltage (V)": 5000.0,
                "pulse_v (V)": 0.0,
                "pulse_l (pJ)": 0.0,
                EGID: gid,
                HASM: has_match,
            })
    return pd.DataFrame(rows)


def test_recovered_atoms_keep_acquisition_order_not_start_counter(monkeypatch):
    monkeypatch.setattr(pr, "load_detector_constants", lambda *a, **k: {"detector_limit_cm": 100.0})

    # Acquisition sequence (TDC stream order):
    #   A matched gid0 sc=100  -> native row 0
    #   B orphan  sc=50        -> 3-channel {0,1,2} -> recovered (after row 0)
    #   C matched gid1 sc=100  -> native row 1   (same sc as A: a counter reuse)
    #   D orphan  sc=30        -> 3-channel {0,1,2} -> recovered (after row 1)
    tdc = _tdc_rows([
        (100, 0, True, [0, 1, 2, 3]),
        (50, -1, False, [0, 1, 2]),
        (100, 1, True, [0, 1, 2, 3]),
        (30, -1, False, [0, 1, 2]),
    ])
    dld = _native_dld(gids=[0, 1], start_counters=[100, 100])
    variables = _FakeVariables(dld, tdc)

    n_recovered = pr.merge_partial_tdc_into_dld(variables, verbose=False)
    assert n_recovered == 2, f"expected 2 recovered atoms, got {n_recovered}"

    merged = variables.data
    assert len(merged) == 4  # 2 native + 2 recovered

    # Correct acquisition order: native0, recovered(B), native1, recovered(D).
    dlts_q = merged[pr.DLTS_QUALITY_COLUMN].tolist()
    assert dlts_q == ["native", "recovered_xy_3of4", "native", "recovered_xy_3of4"], dlts_q
    # Native rows keep their original relative order (markers 0 then 1).
    native_markers = merged.loc[merged[pr.DLTS_QUALITY_COLUMN] == "native", "native_marker"].tolist()
    assert native_markers == [0, 1]
    # A start_counter sort would have put the orphans (30, 50) first; assert
    # the sequence is NOT sorted by start_counter.
    assert merged["start_counter"].tolist() != sorted(merged["start_counter"].tolist())


def test_one_d_partials_excluded_by_default(monkeypatch):
    monkeypatch.setattr(pr, "load_detector_constants", lambda *a, **k: {"detector_limit_cm": 100.0})
    # Orphan pulse with only a complete X axis (ch0, ch1) -> 1-D x partial,
    # which has NaN y and must NOT be merged by default.
    tdc = _tdc_rows([
        (100, 0, True, [0, 1, 2, 3]),
        (50, -1, False, [0, 1]),  # only X axis -> 1-D partial
    ])
    dld = _native_dld(gids=[0], start_counters=[100])
    variables = _FakeVariables(dld, tdc)
    n = pr.merge_partial_tdc_into_dld(variables, verbose=False)
    assert n == 0  # the lone 1-D partial is skipped
    assert len(variables.data) == 1


def test_one_d_partials_included_when_requested(monkeypatch):
    monkeypatch.setattr(pr, "load_detector_constants", lambda *a, **k: {"detector_limit_cm": 100.0})
    tdc = _tdc_rows([
        (100, 0, True, [0, 1, 2, 3]),
        (50, -1, False, [0, 1]),
    ])
    dld = _native_dld(gids=[0], start_counters=[100])
    variables = _FakeVariables(dld, tdc)
    n = pr.merge_partial_tdc_into_dld(variables, include_one_d_partials=True, verbose=False)
    assert n == 1
    merged = variables.data
    # The 1-D partial has a NaN detector coordinate.
    rec = merged[merged[pr.DLTS_QUALITY_COLUMN].str.startswith("recovered")]
    assert len(rec) == 1
    assert bool(rec["y_det (cm)"].isna().any()) or bool(rec["x_det (cm)"].isna().any())
