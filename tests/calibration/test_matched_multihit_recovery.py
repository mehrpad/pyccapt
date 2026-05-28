"""Tests for residual recovery from MATCHED multi-hit pulses (Q3).

The Surface Concept quadrupel finder keeps one hit per pulse. A matched
pulse that fired more stops than its DLD event(s) used can hold a second
ion the firmware discarded. ``recover_from_matched_multihit`` subtracts
the firmware's stops (inverse-matched to its DLD event) and recovers the
residual ion.
"""
import numpy as np
import pandas as pd

from pyccapt.calibration.data_tools import partial_recovery as pr
from pyccapt.calibration.data_tools._raw_workflow_surface_concept import (
    _surface_concept_position_from_pair as _pos,
    _surface_concept_hit_from_time_data as _hit4,
)

EGID = pr.EVENT_GROUP_ID_COLUMN
HASM = pr.TDC_HAS_DLD_MATCH_COLUMN


class _FakeVariables:
    def __init__(self, data, data_tdc):
        self.data = data
        self.data_tdc = data_tdc
        self.data_backup = None
        self.data_tdc_backup = None
        self.max_tof = 1_000_000.0
        self.flight_path_length = 110.0

    def sync_from_data(self, *args, **kwargs):
        return self.data


def _dld_one_event(gid, sc, x_det, y_det, tof=1000.0):
    return pd.DataFrame({
        "x (nm)": [0.0], "y (nm)": [0.0], "z (nm)": [0.0],
        "mc (Da)": [0.0], "mc_uc (Da)": [0.0],
        "high_voltage (V)": [5000.0], "pulse_v (V)": [0.0], "pulse_l (pJ)": [0.0],
        "t (ns)": [tof],
        "x_det (cm)": [x_det], "y_det (cm)": [y_det],
        "start_counter": [sc], EGID: [gid],
    })


def _tdc(channels, times, sc, gid, has_match):
    n = len(channels)
    return pd.DataFrame({
        "channel": np.asarray(channels, dtype=np.int64),
        "start_counter": np.full(n, sc, dtype=np.int64),
        "time_data": np.asarray(times, dtype=np.int64),
        "high_voltage (V)": np.full(n, 5000.0),
        "pulse_v (V)": np.zeros(n),
        "pulse_l (pJ)": np.zeros(n),
        EGID: np.full(n, gid, dtype=np.int64),
        HASM: np.full(n, has_match, dtype=bool),
    })


def test_residual_recovered_from_matched_two_ion_pulse(monkeypatch):
    monkeypatch.setattr(pr, "load_detector_constants", lambda *a, **k: {"detector_limit_cm": 100.0})

    # Ion A (firmware kept) and ion B (residual), well separated in ToF.
    a_t = [100, 140, 130, 110]          # ch0,1,2,3  -> short ToF
    b_t = [2000, 2060, 2050, 2010]      # ch0,1,2,3  -> long ToF (different ion)
    x_a, y_a, tof_a = _hit4(a_t)
    x_b, y_b = _pos(2000, 2060), _pos(2050, 2010)

    tdc = _tdc(
        channels=[0, 1, 2, 3, 0, 1, 2, 3],
        times=a_t + b_t,
        sc=100, gid=0, has_match=True,
    )
    # The native DLD event carries ion A's recorded flight time, which is how
    # the firmware's stops are told apart from ion B's during subtraction.
    dld = _dld_one_event(gid=0, sc=100, x_det=x_a, y_det=y_a, tof=tof_a)
    variables = _FakeVariables(dld, tdc)

    # Off by default: with no orphans and the flag off, nothing is recovered.
    n_off = pr.merge_partial_tdc_into_dld(_FakeVariables(dld.copy(), tdc.copy()), verbose=False)
    assert n_off == 0

    # On: the firmware's stops (ion A) are subtracted, ion B is recovered.
    n = pr.merge_partial_tdc_into_dld(variables, recover_from_matched_multihit=True, verbose=False)
    assert n == 1, f"expected 1 residual atom, got {n}"
    merged = variables.data
    assert len(merged) == 2  # native A + residual B
    rec = merged[merged[pr.DLTS_QUALITY_COLUMN].astype(str).str.startswith("recovered")]
    assert len(rec) == 1
    assert int(rec[pr.DLTS_COLUMN].iloc[0]) == 4  # full xy
    assert abs(float(rec["x_det (cm)"].iloc[0]) - x_b) < 1e-6
    assert abs(float(rec["y_det (cm)"].iloc[0]) - y_b) < 1e-6
    # Order preserved: native A first, residual B second (inserted after A).
    assert merged[pr.DLTS_QUALITY_COLUMN].tolist() == ["native", "recovered_xy"]


def test_single_hit_matched_pulse_yields_no_residual(monkeypatch):
    monkeypatch.setattr(pr, "load_detector_constants", lambda *a, **k: {"detector_limit_cm": 100.0})
    # Exactly 4 stops = one ion; firmware used them all -> nothing residual.
    tdc = _tdc([0, 1, 2, 3], [100, 140, 130, 110], sc=100, gid=0, has_match=True)
    dld = _dld_one_event(gid=0, sc=100, x_det=_pos(100, 140), y_det=_pos(130, 110))
    variables = _FakeVariables(dld, tdc)
    n = pr.merge_partial_tdc_into_dld(variables, recover_from_matched_multihit=True, verbose=False)
    assert n == 0
    assert len(variables.data) == 1


def test_unmatchable_firmware_stops_skip_pulse(monkeypatch):
    monkeypatch.setattr(pr, "load_detector_constants", lambda *a, **k: {"detector_limit_cm": 100.0})
    # The native event's position does not correspond to any stop pair
    # within tolerance -> the pulse is skipped (no double-counting risk).
    tdc = _tdc([0, 1, 2, 3, 0, 1, 2, 3], [100, 140, 130, 110, 200, 260, 250, 210],
               sc=100, gid=0, has_match=True)
    dld = _dld_one_event(gid=0, sc=100, x_det=999.0, y_det=999.0)  # nonsense position
    variables = _FakeVariables(dld, tdc)
    n = pr.merge_partial_tdc_into_dld(
        variables, recover_from_matched_multihit=True, multihit_match_tol_cm=0.05, verbose=False
    )
    assert n == 0


def test_firmware_hit_was_3channel_skips_no_cross_ion_artifact(monkeypatch):
    monkeypatch.setattr(pr, "load_detector_constants", lambda *a, **k: {"detector_limit_cm": 100.0})
    # Firmware's DLD event for ion A was reconstructed from only 3 raw stops
    # (A's ch3 missing from the raw stream); a full ion B is also present.
    # No coherent 4-stop set for A exists, so subtraction must SKIP the pulse
    # rather than stitch A's x to B's y (which share a similar position) and
    # fabricate a garbage residual or double-count A.
    a_t3 = [100, 140, 130]              # A: ch0, ch1, ch2 only (ch3 dropped)
    b_t = [2000, 2060, 2050, 2010]      # B: full, far in ToF
    x_a, y_a, tof_a = _hit4([100, 140, 130, 110])  # firmware inferred A's ch3=110
    tdc = _tdc([0, 1, 2, 0, 1, 2, 3], a_t3 + b_t, sc=100, gid=0, has_match=True)
    dld = _dld_one_event(gid=0, sc=100, x_det=x_a, y_det=y_a, tof=tof_a)
    variables = _FakeVariables(dld, tdc)
    n = pr.merge_partial_tdc_into_dld(variables, recover_from_matched_multihit=True, verbose=False)
    assert n == 0
    # Only the native A row remains; ion B is NOT spuriously recovered here.
    assert len(variables.data) == 1
    assert variables.data[pr.DLTS_QUALITY_COLUMN].tolist() == ["native"]
