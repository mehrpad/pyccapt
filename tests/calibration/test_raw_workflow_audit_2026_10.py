"""Regression tests for the 2026-10 raw-data workflow audit fixes."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.core.share_variables import Variables
# raw_data_workflow first: it resolves the import cycle between the
# _raw_workflow_* sibling modules.
from pyccapt.calibration.data_tools import raw_data_workflow
from pyccapt.calibration.data_tools import (
    _raw_workflow_sc_combinatorial as combinatorial,
    _raw_workflow_surface_concept as sc_workflow,
    data_loadcrop,
    data_tools,
    raw_data_surface_concept,
)
from pyccapt.calibration.leap_tools import ccapt_tools
from pyccapt.calibration.tutorials.tutorials_helpers import (
    auto_raw_analysis_domain,
    helper_auto_raw_analysis,
    helper_data_loader,
)
from pyccapt.calibration.tutorials.tutorials_helpers.helper_roi_concentration_profile import (
    build_roi_concentration_profile_panel,
)

# position (cm) = -0.5 * (t_high - t_low) * XY_FACTOR * 0.1  ->  bins per cm
_BINS_PER_CM = 1.0 / (0.5 * raw_data_workflow.XY_FACTOR * 0.1)


def _pair_for_position(position_cm: float, centre: int = 10000) -> tuple[int, int]:
    """Return (t_low, t_high) whose reconstructed coordinate is ``position_cm``."""
    half = int(round(position_cm * _BINS_PER_CM / 2.0))
    return centre + half, centre - half


def _processed_frame(num_rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x (nm)": np.zeros(num_rows),
            "y (nm)": np.zeros(num_rows),
            "z (nm)": np.zeros(num_rows),
            "mc (Da)": np.linspace(20.0, 30.0, num_rows),
            "mc_uc (Da)": np.linspace(20.0, 30.0, num_rows),
            "high_voltage (V)": np.linspace(1000.0, 4000.0, num_rows),
            "pulse_v (V)": np.full(num_rows, 200.0),
            "pulse_l (pJ)": np.zeros(num_rows),
            "t (ns)": np.linspace(100.0, 600.0, num_rows),
            "t_c (ns)": np.zeros(num_rows),
            "x_det (cm)": np.linspace(-1.0, 1.0, num_rows),
            "y_det (cm)": np.linspace(-1.0, 1.0, num_rows),
            "delta_p": np.zeros(num_rows, dtype=np.uint32),
            "multi": np.ones(num_rows, dtype=np.uint32),
            "start_counter": np.arange(num_rows, dtype=np.uint32),
        }
    )


# --- 1. DLTS classifier keeps repeated sub-events in separate chunks --------


def test_classify_pulse_chunks_multi_hit_pulse_is_two_complete_events():
    tdc = pd.DataFrame({"event_group_id": [0] * 8, "channel": [0, 1, 2, 3, 0, 1, 2, 3]})
    out = auto_raw_analysis_domain._classify_pulse_chunks(tdc, "event_group_id", "surface_concept")
    assert out["complete"].tolist() == [8, 8]
    assert out["partial"].tolist() == []


# --- 2. orphan pulses are grouped by consecutive start_counter runs ----------


def test_dlts_breakdown_does_not_merge_orphan_pulses_across_counter_wraps(monkeypatch):
    # Orphan pulse (sc=5, x pair), matched pulse (sc=6), orphan pulse after a
    # counter wrap with the same value (sc=5, y pair).
    tdc = pd.DataFrame(
        {
            "start_counter": [5, 5, 6, 6, 6, 6, 5, 5],
            "channel": [0, 1, 0, 1, 2, 3, 2, 3],
            "event_group_id": [-1, -1, 0, 0, 0, 0, -1, -1],
            "has_dld_match": [False, False, True, True, True, True, False, False],
        }
    )
    captured: list[str] = []
    monkeypatch.setattr(auto_raw_analysis_domain, "_md", captured.append)
    monkeypatch.setattr(auto_raw_analysis_domain, "display", lambda *_: None)
    auto_raw_analysis_domain.plot_dlts_per_pulse(tdc, "surface_concept")
    plt.close("all")
    text = "\n".join(captured)
    assert "Total pulse triggers: 3" in text
    # Two orphan partials, not one fabricated complete event.
    assert "**4 DLTS** — complete (both delay lines fired): 1 (" in text
    assert "**2 DLTS** — one delay line fired:              2 (" in text


# --- 3. orphan tdc rows outside the kept acquisition span are dropped -------


def _orphan_layout():
    dld_sc = np.array([20, 21, 22])
    tdc_sc = np.array([10, 10, 20, 20, 11, 21, 21, 12, 22, 22, 13])
    dld_gid, tdc_gid, has_match = data_loadcrop.build_event_group_mapping(dld_sc, tdc_sc)
    dld = pd.DataFrame({"start_counter": dld_sc, "event_group_id": dld_gid})
    tdc = pd.DataFrame({"start_counter": tdc_sc, "event_group_id": tdc_gid, "has_dld_match": has_match})
    return dld, tdc


def test_filter_tdc_by_dld_drops_orphans_of_temporally_cropped_region():
    dld, tdc = _orphan_layout()
    middle = data_loadcrop.filter_tdc_by_dld(dld.iloc[[1]], tdc)
    assert middle["start_counter"].tolist() == [21, 21]
    tail = data_loadcrop.filter_tdc_by_dld(dld.iloc[[1, 2]], tdc)
    assert tail["start_counter"].tolist() == [21, 21, 12, 22, 22, 13]
    head = data_loadcrop.filter_tdc_by_dld(dld.iloc[[0]], tdc)
    assert head["start_counter"].tolist() == [10, 10, 20, 20]
    assert len(data_loadcrop.filter_tdc_by_dld(dld, tdc)) == len(tdc)


# --- 4. combinatorial partials store NaN for the missing axis ----------------


def test_combinatorial_partial_hit_has_nan_for_missing_axis_and_finite_mc():
    t_low, t_high = _pair_for_position(1.0)
    record = {"channels": [0, 1], "time_data": [t_low, t_high], "start_counter": [7], "high_voltage": 5000.0, "pulse": 0.0}
    emitted, _ = combinatorial.extract_valid_hits_combinatorial(record, detector_limit_cm=4.0)
    assert len(emitted) == 1
    hit = emitted[0]
    assert hit["detector_axis"] == "x"
    assert hit["x_det (cm)"] == pytest.approx(1.0, abs=1e-3)
    assert np.isnan(hit["y_det (cm)"])
    assert np.isfinite(hit["mc (Da)"])


def test_processed_dataframe_keeps_partial_hit_identity_columns():
    hit_table = pd.DataFrame(
        {
            "start_counter": [1, 2],
            "high_voltage (V)": [5000.0, 5000.0],
            "pulse": [0.0, 0.0],
            "tof (ns)": [100.0, 110.0],
            "mc (Da)": [12.0, 13.0],
            "x_det (cm)": [0.5, 1.0],
            "y_det (cm)": [0.2, np.nan],
            "dlts": [4, 2],
            "detector_axis": ["xy", "x"],
            "in_detector": [True, True],
        }
    )
    processed = raw_data_workflow.surface_concept_hits_to_processed_dataframe(hit_table)
    assert processed["dlts"].tolist() == [4, 2]
    assert processed["detector_axis"].tolist() == ["xy", "x"]


# --- 5. time-sum check, radial detector gate, residual tie-break ------------


def test_normalize_sequence_rejects_crossed_pairing_by_time_sum():
    _, _, _, flags = raw_data_surface_concept._normalize_sequence([1] * 4, [0, 1, 2, 3], [100, 140, 5000, 5040])
    assert flags == [False]
    _, _, _, flags = raw_data_surface_concept._normalize_sequence([1] * 4, [0, 1, 2, 3], [100, 140, 130, 110])
    assert flags == [True]


def test_diagnostics_do_not_accept_flagged_chunk_with_inconsistent_time_sum():
    record = {
        "channels": [0, 1, 2, 3],
        "time_data": [100, 140, 5000, 5040],
        "start_counter": [1],
        "valid_event": [True],
        "high_voltage": 5000.0,
        "pulse": 0.0,
    }
    diagnostics = sc_workflow.build_surface_concept_recovery_diagnostics([record], detector_limit_cm=4.0)
    assert not (diagnostics["dlts"] == 4).any()


def test_complete_hit_detector_gate_is_radial():
    t0, t1 = _pair_for_position(3.5)
    t2, t3 = _pair_for_position(3.5)
    record = {
        "channels": [0, 1, 2, 3],
        "time_data": [t0, t1, t2, t3],
        "start_counter": [1],
        "valid_event": [True],
        "high_voltage": 5000.0,
        "pulse": 0.0,
    }
    diagnostics = sc_workflow.build_surface_concept_recovery_diagnostics([record], detector_limit_cm=4.0)
    assert diagnostics["status"].tolist() == ["4 DLTS outside detector"]
    assert combinatorial._hit_in_detector_axis_aware(3.5, 3.5, "xy", 4.0) is False


def test_off_detector_complete_is_not_split_into_two_partials():
    # (3.5, 3.5) cm: each axis is inside ±4 cm but the radius (4.95 cm) is not.
    t0, t1 = _pair_for_position(3.5)
    t2, t3 = _pair_for_position(3.5)
    record = {"channels": [0, 1, 2, 3], "time_data": [t0, t1, t2, t3], "start_counter": [1],
              "high_voltage": 5000.0, "pulse": 0.0}
    for mode in ("greedy", "exhaustive"):
        emitted, _ = combinatorial.extract_valid_hits_combinatorial(record, mode=mode, detector_limit_cm=4.0)
        assert emitted == []


def test_greedy_selection_breaks_ties_by_pair_sum_residual():
    # Two complete candidates share ch0/ch1/ch2; the first enumerated one
    # (ch3@1150) has a 50-bin time-sum residual, the second (ch3@1100) none.
    record = {
        "channels": [0, 1, 2, 3, 3],
        "time_data": [1000, 1000, 900, 1150, 1100],
        "start_counter": [1],
        "high_voltage": 5000.0,
        "pulse": 0.0,
    }
    emitted, _ = combinatorial.extract_valid_hits_combinatorial(record, mode="greedy", detector_limit_cm=4.0)
    completes = [hit for hit in emitted if hit["dlts"] == 4]
    assert len(completes) == 1
    assert 4 in completes[0]["used_indices"]


# --- 6. real / unavailable combinatorial stats; configured radius -----------


def _sc_frame(channels, times, start_counters):
    n = len(channels)
    return pd.DataFrame(
        {
            "start_counter": start_counters,
            "channel": channels,
            "time_data": times,
            "high_voltage (V)": np.full(n, 5000.0),
            "pulse_v (V)": np.zeros(n),
        }
    )


def test_combinatorial_stats_are_real_or_unavailable_and_use_configured_radius():
    x_low, x_high = _pair_for_position(5.0)
    a0, a1 = _pair_for_position(0.2)
    b0, b1 = _pair_for_position(-0.2, centre=12000)
    frame = _sc_frame(
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
        [a0, a1, a0, a1, b0, b1, b0, b1, x_low, x_high],
        [1] * 8 + [2, 2],
    )
    result = combinatorial.analyze_surface_concept_tdc_frame_combinatorial(frame, detector_limit_cm=10.0)
    assert result["raw_summary"]["multi_hit_groups_of_four_timestamps"] == 8
    assert result["recovery_stats"]["outside_detector_hits"] is None
    assert result["recovery_stats"]["unrecoverable_chunks"] is None
    # The 5 cm x-partial is inside the configured 10 cm radius.
    assert result["sequence_stats"]["dld2"] == {2: 1}
    fig = sc_workflow.plot_surface_concept_recovery_summary(result["recovery_stats"])
    assert fig is not None
    plt.close(fig)


def test_sequence_summary_uses_configured_detector_radius():
    x_low, x_high = _pair_for_position(5.0)
    records = [{"channels": [0, 1], "time_data": [x_low, x_high], "valid_event": [False]}]
    assert raw_data_workflow.summarize_surface_concept_sequences(records, detector_limit_cm=10.0)["dld2"] == {2: 1}
    assert raw_data_workflow.summarize_surface_concept_sequences(records, detector_limit_cm=4.0)["dld2"] == {}


# --- 7. spatial crop drops single-axis partials outside the circle ----------


def test_spatial_crop_drops_partials_whose_known_axis_is_outside():
    frame = pd.DataFrame(
        {
            "x_det (cm)": [-1.0, 0.0, 1.0, 0.1, 3.0, np.nan, np.nan, np.nan],
            "y_det (cm)": [-1.0, 0.0, 1.0, np.nan, np.nan, 0.2, -3.0, np.nan],
            "row": np.arange(8),
        }
    )
    variables = SimpleNamespace(selected_x_fdm=0.0, selected_y_fdm=0.0, roi_fdm=0.5)
    cropped = data_loadcrop.crop_data_after_selection(frame, variables)
    assert cropped["row"].tolist() == [1, 3, 5, 7]


# --- 8. APT start_counter is never fabricated from tElapsed -----------------


def test_apt_start_counter_unavailable_without_pulse_number(monkeypatch):
    raw = pd.DataFrame({"Mass": [27.0, 54.0], "x": [1.0, 2.0], "y": [0.0, 0.0], "z": [1.0, 2.0], "tElapsed": [0.4, 1.7]})
    monkeypatch.setattr(ccapt_tools.leap_tools, "read_apt", lambda _: raw.copy())
    result = ccapt_tools.apt_to_ccapt("dummy.apt")
    assert "start_counter" in result.attrs["pyccapt_unavailable_columns"]
    assert result["start_counter"].tolist() == [0, 0]


# --- 9. ROI panel defaults follow the latest reconstruction ------------------


def test_roi_panel_defaults_read_coordinates_at_use_time():
    ranges = pd.DataFrame(
        {"name": ["A"], "mc_low": [0.5], "mc_up": [1.5], "element": [["A"]], "complex": [[1]]}
    )
    variables = SimpleNamespace(x=np.array([]), y=np.array([]), z=np.array([]), mc=np.array([]),
                                range_data=ranges, result_path="")
    panel = build_roi_concentration_profile_panel(variables)
    axis_widget = panel.children[1].children[1]
    profile_start = panel.children[6].children[1]
    profile_end = panel.children[7].children[1]
    variables.x = np.array([10.0, 20.0, 30.0])
    variables.y = np.array([1.0, 2.0, 3.0])
    variables.z = np.array([-5.0, 0.0, 5.0])
    axis_widget.value = "x"
    assert (profile_start.value, profile_end.value) == (10.0, 30.0)


# --- 10. failed partial-hit merge is visible in the loader status -----------


def test_load_data_reports_failed_partial_merge(tmp_path, monkeypatch):
    from pyccapt.calibration.data_tools import partial_recovery

    dataset_path = tmp_path / "sample.h5"
    tdc_df = pd.DataFrame({"start_counter": [0, 0], "channel": [0, 1], "time_data": [1, 2]})
    monkeypatch.setattr(
        helper_data_loader.data_tools, "load_data", lambda *a, **k: (_processed_frame(), tdc_df.copy())
    )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("synthetic merge failure")

    monkeypatch.setattr(partial_recovery, "merge_partial_tdc_into_dld", _boom)
    variables = Variables()
    with pytest.warns(RuntimeWarning, match="merge_partial_tdc failed"):
        helper_data_loader.load_data(
            str(dataset_path), 120.0, 110.0, "voltage", "pyccapt", variables,
            processing_mode=False, load_tdc_raw=True, show_progress=False,
        )
    assert variables.partial_merge_status.startswith("failed")
    assert "synthetic merge failure" in variables.partial_merge_status


# --- 11. tdc layout fallback is explicit -------------------------------------


def test_raw_tdc_layout_fallback_warns(monkeypatch):
    def _fetch(path, tdc_extract_mode):
        if tdc_extract_mode == "tdc_sc":
            raise KeyError("no SC layout")
        return "dld", "tdc"

    monkeypatch.setattr(data_loadcrop, "fetch_dataset_with_tdc", _fetch)
    with pytest.warns(RuntimeWarning, match="tdc_extract_mode='tdc_ro'"):
        result = data_tools.load_data("dummy.h5", "pyccapt", mode="raw", load_tdc=True)
    assert result == ("dld", "tdc")


# --- 12. auto raw analysis uses the available t0 and says mc is uncalibrated -


def test_resolve_raw_t0_prefers_variables_then_metadata():
    frame = pd.DataFrame({"t (ns)": [1.0]})
    assert helper_auto_raw_analysis.resolve_raw_t0_ns(SimpleNamespace(t0=12.5), frame)[0] == 12.5
    value, source = helper_auto_raw_analysis.resolve_raw_t0_ns(SimpleNamespace(), frame)
    assert value == 0.0 and "not available" in source


def test_run_analysis_passes_t0_and_states_mc_is_uncalibrated(monkeypatch):
    class _Stop(Exception):
        pass

    captured: dict = {}

    def _fake_combinatorial(_tdc, **kwargs):
        captured.update(kwargs)
        raise _Stop

    shown: list[str] = []
    monkeypatch.setattr(raw_data_workflow, "analyze_surface_concept_tdc_frame_combinatorial", _fake_combinatorial)
    monkeypatch.setattr(helper_auto_raw_analysis, "display", lambda obj: shown.append(getattr(obj, "data", str(obj))))
    tdc = pd.DataFrame({"channel": [0, 1, 2, 3], "start_counter": [1] * 4, "time_data": [1, 2, 3, 4],
                        "high_voltage (V)": [5000.0] * 4, "pulse_v (V)": [0.0] * 4})
    variables = SimpleNamespace(data=_processed_frame(), data_tdc=tdc, dataset_name="x", path="", t0=12.5,
                                flight_path_length=110.0, pulse_mode="voltage", conf=None)
    species = [{"label": "Al", "mc_low": 26.5, "mc_up": 27.5}]
    with pytest.raises(_Stop):
        helper_auto_raw_analysis.run_analysis(variables, species, peak_units="mc")
    assert captured["t0"] == 12.5
    text = "\n".join(str(item) for item in shown)
    assert "No voltage or bowl calibration" in text
    assert "uncalibrated hit-table mc" in text
