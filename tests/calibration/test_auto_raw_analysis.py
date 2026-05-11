"""Unit tests for the auto-raw-analysis helpers and the save_range round-trip.

These tests deliberately avoid invoking matplotlib pipelines: they cover the
pure data plumbing (detector detection, species extraction from a range
table, save_data with save_range, and the bundled load helper).
"""
from __future__ import annotations

from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.core.share_variables import Variables
from pyccapt.calibration.data_tools import data_loadcrop, data_tools
from pyccapt.calibration.tutorials.tutorials_helpers import (
    helper_auto_raw_analysis,
    helper_data_loader,
)


# ---------------------------------------------------------------------------
# Detector detection
# ---------------------------------------------------------------------------


def test_detect_detector_kind_surface_concept():
    tdc = pd.DataFrame({"channel": [0, 1, 2, 3, 0, 1, 2, 3]})
    assert helper_auto_raw_analysis.detect_detector_kind(tdc) == "surface_concept"


def test_detect_detector_kind_roentdek():
    tdc = pd.DataFrame({"channel": [0, 1, 2, 3, 4, 5]})
    assert helper_auto_raw_analysis.detect_detector_kind(tdc) == "roentdek"


def test_detect_detector_kind_single_delay_line():
    """A TDC stream with channels only in {0, 1} corresponds to a 1-DL detector."""
    tdc = pd.DataFrame({"channel": [0, 1, 0, 1, 1, 0]})
    assert helper_auto_raw_analysis.detect_detector_kind(tdc) == "single_delay_line"


def test_detect_detector_kind_unknown_for_empty_or_missing():
    assert helper_auto_raw_analysis.detect_detector_kind(None) == "unknown"
    assert helper_auto_raw_analysis.detect_detector_kind(pd.DataFrame()) == "unknown"


def test_expected_dlts_full():
    assert helper_auto_raw_analysis.expected_dlts_full("single_delay_line") == 2
    assert helper_auto_raw_analysis.expected_dlts_full("surface_concept") == 4
    assert helper_auto_raw_analysis.expected_dlts_full("roentdek") == 6
    assert helper_auto_raw_analysis.expected_dlts_full("unknown") == 0


def test_delay_line_pairs():
    """The (low, high) channel pairs that drive the chunked DLTS classifier."""
    assert helper_auto_raw_analysis._delay_line_pairs("single_delay_line") == [(0, 1)]
    assert helper_auto_raw_analysis._delay_line_pairs("surface_concept") == [(0, 1), (2, 3)]
    assert helper_auto_raw_analysis._delay_line_pairs("roentdek") == [(0, 1), (2, 3), (4, 5)]
    assert helper_auto_raw_analysis._delay_line_pairs("unknown") == []


# ---------------------------------------------------------------------------
# Species table extraction
# ---------------------------------------------------------------------------


def test_species_from_range_skips_unranged_and_invalid():
    range_df = pd.DataFrame({
        "name": ["Al", "unranged0", "Cr", "bad"],
        "ion": ["$Al^+$", "un", "$Cr^+$", "X"],
        "mc_low": [26.78, 0.0, 51.79, 5.0],
        "mc_up":  [27.18, 400.0, 52.19, 4.0],   # last row: invalid (up <= low)
        "color":  ["#aaa", "#000", "#bbb", "#ccc"],
    })
    species = helper_auto_raw_analysis.species_from_range(range_df)
    labels = [s["label"] for s in species]
    assert labels == ["Al", "Cr"]
    assert species[0]["mc_low"] == pytest.approx(26.78)
    assert species[0]["mc_up"] == pytest.approx(27.18)
    assert species[0]["color"] == "#aaa"


def test_species_from_range_handles_none_or_empty():
    assert helper_auto_raw_analysis.species_from_range(None) == []
    assert helper_auto_raw_analysis.species_from_range(pd.DataFrame()) == []


def test_compute_mrp_half_returns_finite_for_clean_peak():
    rng = np.random.default_rng(0)
    peak = rng.normal(loc=27.0, scale=0.05, size=2000)
    mrp = helper_auto_raw_analysis.compute_mrp_half(peak)
    assert np.isfinite(mrp)
    assert mrp > 0


def test_compute_mrp_half_returns_nan_for_tiny_window():
    assert np.isnan(helper_auto_raw_analysis.compute_mrp_half(np.array([27.0, 27.1])))


# ---------------------------------------------------------------------------
# save_data with save_range round-trip
# ---------------------------------------------------------------------------


class _StubVariables:
    """Minimal stand-in for `Variables` for save tests."""

    def __init__(self, output_dir: Path, name: str = "calibrated"):
        self._output_dir = output_dir
        self.result_data_name = name
        self.result_data_path = str(output_dir) + "/"
        self.result_path = str(output_dir) + "/"
        self.data_tdc = None
        self.range_data = None

    def resolve_result_data_file(self, filename: str) -> str:
        return str(self._output_dir / filename)

    def resolve_result_file(self, filename: str) -> str:
        return str(self._output_dir / filename)


def _make_simple_dld(num_rows: int = 6) -> pd.DataFrame:
    return pd.DataFrame({
        "high_voltage (V)": np.linspace(1000.0, 4000.0, num_rows),
        "pulse_v (V)": np.full(num_rows, 200.0),
        "pulse_l (pJ)": np.zeros(num_rows),
        "t (ns)": np.linspace(100.0, 600.0, num_rows),
        "x_det (cm)": np.linspace(-1.0, 1.0, num_rows),
        "y_det (cm)": np.linspace(-1.0, 1.0, num_rows),
        "mc (Da)": np.linspace(20.0, 30.0, num_rows),
        "mc_uc (Da)": np.linspace(20.0, 30.0, num_rows),
        "x (nm)": np.zeros(num_rows),
        "y (nm)": np.zeros(num_rows),
        "z (nm)": np.zeros(num_rows),
        "t_c (ns)": np.zeros(num_rows),
        "delta_p": np.zeros(num_rows, dtype=np.uint32),
        "multi": np.ones(num_rows, dtype=np.uint32),
        "start_counter": np.arange(num_rows, dtype=np.uint32),
    })


def test_save_data_with_save_range_writes_range_group(tmp_path: Path):
    dld_df = _make_simple_dld()
    range_df = pd.DataFrame({
        "name": ["Al", "Cr"],
        "ion": ["$Al^+$", "$Cr^+$"],
        "mass": [26.98, 51.99],
        "mc": [26.98, 51.99],
        "mc_low": [26.78, 51.79],
        "mc_up": [27.18, 52.19],
        "color": ["#aaa", "#bbb"],
        "element": [["Al"], ["Cr"]],
        "complex": [[1], [1]],
        "isotope": [[0], [0]],
        "charge": [1, 1],
    })
    variables = _StubVariables(tmp_path, name="testset")
    variables.range_data = range_df

    data_tools.save_data(dld_df, variables, hdf=True, save_range=True)

    out_path = tmp_path / "testset.h5"
    assert out_path.exists()
    loaded_range = pd.read_hdf(out_path, key="range")
    assert list(loaded_range["name"]) == ["Al", "Cr"]


def test_save_data_warns_when_save_range_but_table_empty(tmp_path: Path):
    dld_df = _make_simple_dld()
    variables = _StubVariables(tmp_path, name="testset")
    variables.range_data = pd.DataFrame()

    with pytest.warns(RuntimeWarning, match="range_data is empty"):
        data_tools.save_data(dld_df, variables, hdf=True, save_range=True)


# ---------------------------------------------------------------------------
# load_calibrated_h5 round-trip
# ---------------------------------------------------------------------------


def test_load_calibrated_h5_round_trip_with_tdc_and_range(tmp_path: Path):
    dld_df = _make_simple_dld(num_rows=4)
    dld_df["event_group_id"] = np.array([0, 1, 2, 3], dtype=np.int64)
    tdc_df = pd.DataFrame({
        "channel": np.arange(8, dtype=np.uint32) % 4,
        "start_counter": np.repeat(np.arange(4), 2).astype(np.uint32),
        "high_voltage (V)": np.full(8, 1500.0),
        "pulse_v (V)": np.full(8, 200.0),
        "pulse_l (pJ)": np.zeros(8),
        "time_data": np.linspace(40.0, 80.0, 8),
        "event_group_id": np.repeat(np.arange(4), 2).astype(np.int64),
        "has_dld_match": np.ones(8, dtype=bool),
    })
    range_df = pd.DataFrame({
        "name": ["Al"],
        "ion": ["$Al^+$"],
        "mass": [26.98],
        "mc": [26.98],
        "mc_low": [26.78],
        "mc_up": [27.18],
        "color": ["#aaa"],
        "element": [["Al"]],
        "complex": [[1]],
        "isotope": [[0]],
        "charge": [1],
    })

    h5_path = tmp_path / "bundled.h5"
    dld_df.to_hdf(h5_path, key="df", mode="w")
    tdc_df.to_hdf(h5_path, key="tdc", mode="a")
    range_df.to_hdf(h5_path, key="range", mode="a")

    variables = Variables()
    loaded_dld, loaded_tdc, loaded_range = helper_data_loader.load_calibrated_h5(
        str(h5_path), variables
    )

    assert len(loaded_dld) == 4
    assert loaded_tdc is not None and len(loaded_tdc) == 8
    assert loaded_range is not None and list(loaded_range["name"]) == ["Al"]
    assert variables.dataset_name == "bundled"
    # The unified loader honors the save-time bundle: dld + tdc + range all populated.
    assert variables.data is not None
    assert variables.data_tdc is not None
    assert len(variables.range_data) == 1


def test_load_calibrated_h5_works_without_tdc_and_range(tmp_path: Path):
    dld_df = _make_simple_dld(num_rows=3)
    h5_path = tmp_path / "dld_only.h5"
    dld_df.to_hdf(h5_path, key="df", mode="w")

    variables = Variables()
    loaded_dld, loaded_tdc, loaded_range = helper_data_loader.load_calibrated_h5(
        str(h5_path), variables
    )

    assert len(loaded_dld) == 3
    assert loaded_tdc is None
    assert loaded_range is None
    assert variables.data_tdc is None


def test_load_calibrated_h5_dispatches_rhit_files_to_leap_loader(tmp_path: Path, monkeypatch):
    """A LEAP CAMECA RHIT file is a Cameca ROOT bundle, not HDF5 — calling
    pd.read_hdf on it raises HDF5ExtError ("file signature not found") which
    is neither ``KeyError`` nor ``ValueError`` and therefore escapes the
    calibrated→raw fallback. The loader must dispatch on the ``.rhit``
    extension before that point and route to the LEAP RHIT loader.

    This was the user-reported bug: ``R56_09048.RHIT`` blew up with
    ``HDF5ExtError`` in the Loading-PyCCAPT-HDF5 progress block.
    """
    rhit_path = tmp_path / "R56_09048.RHIT"
    rhit_path.write_bytes(b"placeholder-rhit-bytes-not-actually-root")

    fake_hits = pd.DataFrame({
        "mc":      [12.0, 27.0, 56.0],
        "tof":     [200.0, 400.0, 600.0],
        "VDC":     [4500.0, 4500.0, 4500.0],
        "detx":    [1.0, -2.0, 0.5],
        "dety":    [0.5, 1.5, -1.0],
        "tElapsed": [0.0, 1.0, 2.0],
        "pulse":   [0.0, 0.0, 0.0],
    })

    rhit_load_calls: list[str] = []

    def fake_rhit_load(path):
        rhit_load_calls.append(str(path))
        return fake_hits, {}, {"format": "RHIT (Cameca ROOT)"}

    # The RHIT branch imports its loader lazily, so we patch the module.
    from pyccapt.calibration.leap_tools import cameca_raw as cameca_raw_pkg

    monkeypatch.setattr(cameca_raw_pkg, "rhit_load", fake_rhit_load)

    variables = Variables()
    loaded_dld, loaded_tdc, loaded_range = helper_data_loader.load_calibrated_h5(
        str(rhit_path), variables, show_progress=False
    )

    # The LEAP RHIT loader was actually called (we don't pretend to read the
    # bytes — just assert that the helper dispatched on extension).
    assert rhit_load_calls == [str(rhit_path)]

    # The returned dld dataframe is the rhit_to_ccapt conversion: it has the
    # processed-schema columns the rest of the analysis pipeline expects.
    assert loaded_dld is not None
    assert len(loaded_dld) == 3
    expected_cols = {
        "x (nm)", "y (nm)", "z (nm)",
        "mc (Da)", "mc_uc (Da)",
        "high_voltage (V)", "pulse_v (V)", "pulse_l (pJ)",
        "t (ns)", "t_c (ns)",
        "x_det (cm)", "y_det (cm)",
        "delta_p", "multi", "start_counter",
    }
    assert expected_cols.issubset(loaded_dld.columns)
    # Values come straight from the fake hits (with the cm = mm/10 conversion).
    assert list(loaded_dld["mc (Da)"]) == [12.0, 27.0, 56.0]
    assert list(loaded_dld["t (ns)"]) == [200.0, 400.0, 600.0]
    assert list(loaded_dld["x_det (cm)"]) == [0.1, -0.2, 0.05]

    # RHIT files have no raw delay-line tdc data and no /range group.
    assert loaded_tdc is None
    assert loaded_range is None
    assert variables.data_tdc is None


def test_load_calibrated_h5_accepts_show_progress_flag(tmp_path: Path):
    dld_df = _make_simple_dld(num_rows=2)
    h5_path = tmp_path / "dld_only_no_progress.h5"
    dld_df.to_hdf(h5_path, key="df", mode="w")

    variables = Variables()
    loaded_dld, loaded_tdc, loaded_range = helper_data_loader.load_calibrated_h5(
        str(h5_path), variables, show_progress=False
    )

    assert len(loaded_dld) == 2
    assert loaded_tdc is None
    assert loaded_range is None


@pytest.mark.parametrize("data_mode,suffix", [("leap_epos", ".epos"), ("leap_apt", ".apt")])
def test_helper_load_data_supports_leap_processing_modes(tmp_path: Path, monkeypatch, data_mode: str, suffix: str):
    dataset_path = tmp_path / f"sample{suffix}"
    dataset_path.write_bytes(b"placeholder")

    source = _make_simple_dld(num_rows=3)

    def fake_load_data(path, mode_name, *args, **kwargs):
        assert path == str(dataset_path)
        assert mode_name == data_mode
        return source.copy()

    monkeypatch.setattr(helper_data_loader.data_tools, "load_data", fake_load_data)

    variables = Variables()
    helper_data_loader.load_data(
        str(dataset_path),
        max_mc=120.0,
        flightPathLength=110.0,
        pulse_mode="voltage",
        tdc=data_mode,
        variables=variables,
        processing_mode=True,
        load_tdc_raw=False,
    )

    assert variables.dataset_name == "sample"
    assert variables.path == str(dataset_path)
    assert variables.data is not None
    assert len(variables.data) == 3
    assert variables.data_tdc is None


def test_load_calibrated_h5_falls_back_to_pure_raw_acquisition_layout(tmp_path: Path):
    """When the file has /dld + /tdc but no /df, the loader should switch to
    the raw acquisition path (parsing /dld with fetch_dataset_with_tdc and
    converting to the processed schema)."""
    import h5py

    dld_sc = np.array([1, 2, 3])
    tdc_sc = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    h5_path = tmp_path / "raw_only.h5"
    n_dld = len(dld_sc)
    n_tdc = len(tdc_sc)
    with h5py.File(h5_path, "w") as hdf:
        dld_grp = hdf.create_group("dld")
        dld_grp.create_dataset("high_voltage", data=np.full((n_dld, 1), 1500.0))
        dld_grp.create_dataset("voltage_pulse", data=np.full((n_dld, 1), 200.0))
        dld_grp.create_dataset("laser_intensity", data=np.zeros((n_dld, 1)))
        dld_grp.create_dataset("start_counter", data=dld_sc.reshape(-1, 1).astype(np.int64))
        dld_grp.create_dataset("t", data=np.full((n_dld, 1), 400.0))
        dld_grp.create_dataset("x", data=np.full((n_dld, 1), 0.5))
        dld_grp.create_dataset("y", data=np.full((n_dld, 1), -0.5))

        tdc_grp = hdf.create_group("tdc")
        tdc_grp.create_dataset("channel", data=(np.arange(n_tdc) % 4).reshape(-1, 1).astype(np.int64))
        tdc_grp.create_dataset("start_counter", data=tdc_sc.reshape(-1, 1).astype(np.int64))
        tdc_grp.create_dataset("high_voltage", data=np.full((n_tdc, 1), 1500.0))
        tdc_grp.create_dataset("voltage_pulse", data=np.full((n_tdc, 1), 200.0))
        tdc_grp.create_dataset("laser_pulse", data=np.zeros((n_tdc, 1)))
        tdc_grp.create_dataset("time_data", data=np.linspace(40.0, 80.0, n_tdc).reshape(-1, 1))

    variables = Variables()
    loaded_dld, loaded_tdc, _ = helper_data_loader.load_calibrated_h5(
        str(h5_path), variables
    )

    # The /df-style processed schema must exist on the loaded dld.
    assert "mc (Da)" in loaded_dld.columns
    assert "t (ns)" in loaded_dld.columns
    assert "event_group_id" in loaded_dld.columns
    assert loaded_tdc is not None
    assert "event_group_id" in loaded_tdc.columns


def test_call_auto_raw_data_analysis_renders_single_panel_with_dropdown():
    """The redesigned UI is one VBox panel (not a Tab) with a peak-source
    dropdown. Toggling to 'range' must disable the manual rows; toggling
    back to 'manual' must re-enable them."""
    import ipywidgets as widgets

    class _CapturingVariables(Variables):
        def __init__(self):
            super().__init__()

    variables = _CapturingVariables()
    variables.data = pd.DataFrame({
        "mc (Da)": [27.0, 27.05, 27.1],
        "t (ns)": [400.0, 410.0, 420.0],
        "x_det (cm)": [0.0, 0.1, -0.1],
        "y_det (cm)": [0.0, 0.1, -0.1],
        "delta_p": [0, 1, 2],
        "multi": [1, 1, 1],
        "start_counter": [1, 2, 3],
    })

    captured = {}
    real_display = helper_auto_raw_analysis.display

    def capture_display(obj):
        captured["panel"] = obj

    helper_auto_raw_analysis.display = capture_display
    try:
        helper_auto_raw_analysis.call_auto_raw_data_analysis(variables)
    finally:
        helper_auto_raw_analysis.display = real_display

    panel = captured["panel"]
    assert isinstance(panel, widgets.VBox)

    # The panel exposes four dropdowns: peak source, peak units, save plots, recovery.
    dropdowns = [c for c in panel.children if isinstance(c, widgets.Dropdown)]
    assert len(dropdowns) == 4
    by_desc = {d.description: d for d in dropdowns}

    peak_source = by_desc["Peak source:"]
    assert {value for _label, value in peak_source.options} == {"manual", "range"}

    save_plots = by_desc["Save plots:"]
    assert save_plots.value is False
    assert {value for _label, value in save_plots.options} == {True, False}

    # Manual rows: a VBox with exactly six child HBoxes.
    manual_grids = [
        c for c in panel.children
        if isinstance(c, widgets.VBox) and len(c.children) == 6
        and all(isinstance(row, widgets.HBox) for row in c.children)
    ]
    assert len(manual_grids) == 1
    manual_grid = manual_grids[0]

    def _all_disabled():
        return all(
            child.children[0].disabled
            and child.children[1].disabled
            and child.children[2].disabled
            for child in manual_grid.children
        )

    # No range data was loaded, so the dropdown defaults to "manual" and rows are enabled.
    assert peak_source.value == "manual"
    assert not _all_disabled()

    # Toggling to "range" must disable the rows.
    peak_source.value = "range"
    assert _all_disabled()

    # Toggling back to "manual" re-enables them.
    peak_source.value = "manual"
    assert not _all_disabled()


def test_run_analysis_saves_plots_beside_dataset(tmp_path: Path):
    variables = Variables()
    dataset_path = tmp_path / "sample.h5"
    dataset_path.touch()
    variables.path = str(dataset_path)
    variables.dataset_name = dataset_path.stem
    variables.data = pd.DataFrame({
        "mc (Da)": [27.0, 27.05, 27.1, 52.0],
        "mc_uc (Da)": [27.0, 27.05, 27.1, 52.0],
        "t (ns)": [400.0, 410.0, 420.0, 510.0],
        "x_det (cm)": [0.0, 0.1, -0.1, 0.05],
        "y_det (cm)": [0.0, 0.1, -0.1, -0.05],
        "delta_p": [0, 1, 2, 3],
        "multi": [1, 1, 2, 1],
        "start_counter": [1, 2, 3, 4],
    })
    variables.data_tdc = None

    species = [{"label": "Al", "mc_low": 26.8, "mc_up": 27.2, "color": "#1f77b4"}]
    real_display = helper_auto_raw_analysis.display
    helper_auto_raw_analysis.display = lambda _obj: None
    try:
        helper_auto_raw_analysis.run_analysis(variables, species, save_plots=True)
    finally:
        helper_auto_raw_analysis.display = real_display

    save_dir = tmp_path / "sample_raw_analysis_plots"
    assert save_dir.is_dir()
    for stem in ("tof_histogram", "mc_histogram", "fdm_all", "fdm_species", "multihit_deadzone"):
        assert (save_dir / f"{stem}.png").is_file()
        assert (save_dir / f"{stem}.svg").is_file()


def test_run_analysis_handles_roentdek_tdc_bundle():
    variables = Variables()
    variables.dataset_name = "roentdek"
    variables.data = pd.DataFrame(
        {
            "event_group_id": [0, 1],
            "start_counter": [10, 11],
            "mc (Da)": [27.0, 54.0],
            "mc_uc (Da)": [27.0, 54.0],
            "t (ns)": [400.0, 500.0],
            "x_det (cm)": [0.1, 0.2],
            "y_det (cm)": [0.3, 0.4],
            "high_voltage (V)": [1000.0, 1100.0],
            "pulse_v (V)": [10.0, 12.0],
            "delta_p": [0, 1],
            "multi": [1, 1],
        }
    )
    variables.data_tdc = pd.DataFrame(
        {
            "channel": np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3], dtype=np.uint32),
            "start_counter": np.array([10, 10, 10, 10, 10, 10, 11, 11, 11, 11], dtype=np.uint32),
            "event_group_id": np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
            "high_voltage (V)": np.array([1000.0] * 6 + [1100.0] * 4, dtype=float),
            "pulse_v (V)": np.array([10.0] * 6 + [12.0] * 4, dtype=float),
        }
    )

    real_display = helper_auto_raw_analysis.display
    helper_auto_raw_analysis.display = lambda _obj: None
    try:
        helper_auto_raw_analysis.run_analysis(
            variables,
            [{"label": "Peak 1", "mc_low": 26.5, "mc_up": 27.5, "color": "#1f77b4"}],
            save_plots=False,
        )
    finally:
        helper_auto_raw_analysis.display = real_display


# ---------------------------------------------------------------------------
# event_group_id ride-through with the new flag combo
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# species_from_manual: validation of the manual-tab widget grid
# ---------------------------------------------------------------------------


def _row(label: str, low: float, high: float):
    return (
        widgets.Text(value=label),
        widgets.FloatText(value=low),
        widgets.FloatText(value=high),
    )


def test_species_from_manual_skips_zero_rows_and_keeps_valid_ones():
    rows = [
        _row("Al", 26.78, 27.18),
        _row("", 0.0, 0.0),                # both zero -> skipped
        _row("Cr", 51.79, 52.19),
    ]
    species = helper_auto_raw_analysis.species_from_manual(rows)
    labels = [s["label"] for s in species]
    assert labels == ["Al", "Cr"]
    assert species[0]["mc_low"] == pytest.approx(26.78)
    assert species[1]["mc_up"] == pytest.approx(52.19)


def test_species_from_manual_uses_default_label_for_blank_label():
    rows = [_row("   ", 10.0, 11.0)]
    species = helper_auto_raw_analysis.species_from_manual(rows)
    assert species[0]["label"] == "Peak 1"


def test_species_from_manual_rejects_inverted_range():
    rows = [_row("X", 5.0, 4.0)]
    with pytest.raises(ValueError, match="max must be greater than min"):
        helper_auto_raw_analysis.species_from_manual(rows)


# ---------------------------------------------------------------------------
# pyccapt_raw_to_processed must propagate event_group_id when present
# ---------------------------------------------------------------------------


def test_pyccapt_raw_to_processed_carries_event_group_id():
    raw = pd.DataFrame({
        "high_voltage (V)": [1000.0, 1100.0],
        "pulse_v (V)": [200.0, 200.0],
        "pulse_l (pJ)": [0.0, 0.0],
        "t (ns)": [400.0, 500.0],
        "x_det (cm)": [0.5, -0.5],
        "y_det (cm)": [0.5, -0.5],
        "start_counter": [11, 12],
        "event_group_id": [42, 43],
    })
    processed = data_tools.pyccapt_raw_to_processed(raw)
    assert "event_group_id" in processed.columns
    assert processed["event_group_id"].tolist() == [42, 43]


def test_pyccapt_raw_to_processed_omits_event_group_id_when_absent():
    raw = pd.DataFrame({
        "high_voltage (V)": [1000.0],
        "pulse_v (V)": [200.0],
        "pulse_l (pJ)": [0.0],
        "t (ns)": [400.0],
        "x_det (cm)": [0.5],
        "y_det (cm)": [0.5],
        "start_counter": [11],
    })
    processed = data_tools.pyccapt_raw_to_processed(raw)
    assert "event_group_id" not in processed.columns


# ---------------------------------------------------------------------------
# load_calibrated_h5: fallback paths for the range table
# ---------------------------------------------------------------------------


def _write_dld_only(h5_path: Path, num_rows: int = 3) -> None:
    _make_simple_dld(num_rows=num_rows).to_hdf(h5_path, key="df", mode="w")


def _write_external_range(range_path: Path) -> None:
    pd.DataFrame({
        "name": ["Al"], "ion": ["$Al^+$"], "mass": [26.98], "mc": [26.98],
        "mc_low": [26.78], "mc_up": [27.18], "color": ["#aaa"],
        "element": [["Al"]], "complex": [[1]], "isotope": [[0]], "charge": [1],
    }).to_hdf(range_path, key="df", mode="w")


def test_load_calibrated_h5_uses_explicit_range_path(tmp_path: Path):
    h5_path = tmp_path / "dld_only.h5"
    _write_dld_only(h5_path)
    external_range = tmp_path / "external_range.h5"
    _write_external_range(external_range)

    variables = Variables()
    _, _, loaded_range = helper_data_loader.load_calibrated_h5(
        str(h5_path), variables, range_path=str(external_range)
    )
    assert loaded_range is not None
    assert list(loaded_range["name"]) == ["Al"]
    assert variables.range_data is not None


def test_load_calibrated_h5_falls_back_to_sibling_range_h5(tmp_path: Path):
    h5_path = tmp_path / "dataset.h5"
    sibling = tmp_path / "dataset_range.h5"
    _write_dld_only(h5_path)
    _write_external_range(sibling)

    variables = Variables()
    _, _, loaded_range = helper_data_loader.load_calibrated_h5(str(h5_path), variables)
    assert loaded_range is not None
    assert list(loaded_range["name"]) == ["Al"]


def test_save_data_save_tdc_and_save_range_in_same_call(tmp_path: Path):
    """All three groups (/df, /tdc, /range) end up in one h5."""
    dld_sc = np.array([1, 2, 3])
    tdc_sc = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    dld_gid, tdc_gid, has_match = data_loadcrop.build_event_group_mapping(dld_sc, tdc_sc)
    dld_df = _make_simple_dld(num_rows=3)
    dld_df["start_counter"] = dld_sc
    dld_df["event_group_id"] = dld_gid
    tdc_df = pd.DataFrame({
        "channel": np.arange(len(tdc_sc), dtype=np.uint32) % 4,
        "start_counter": tdc_sc.astype(np.uint32),
        "high_voltage (V)": np.full(len(tdc_sc), 1500.0),
        "pulse_v (V)": np.full(len(tdc_sc), 200.0),
        "pulse_l (pJ)": np.zeros(len(tdc_sc)),
        "time_data": np.linspace(40.0, 80.0, len(tdc_sc)),
        "event_group_id": tdc_gid,
        "has_dld_match": has_match,
    })
    range_df = pd.DataFrame({
        "name": ["Al"], "ion": ["$Al^+$"], "mass": [26.98], "mc": [26.98],
        "mc_low": [26.78], "mc_up": [27.18], "color": ["#aaa"],
        "element": [["Al"]], "complex": [[1]], "isotope": [[0]], "charge": [1],
    })
    variables = _StubVariables(tmp_path, name="bundle")
    variables.data_tdc = tdc_df
    variables.range_data = range_df

    data_tools.save_data(dld_df, variables, hdf=True, save_tdc=True, save_range=True)

    out_path = tmp_path / "bundle.h5"
    assert pd.read_hdf(out_path, key="df").shape[0] == 3
    assert pd.read_hdf(out_path, key="tdc").shape[0] == len(tdc_sc)
    assert pd.read_hdf(out_path, key="range").shape[0] == 1


# ---------------------------------------------------------------------------
# Chunked DLTS classifier — must match Figure 9 of the PyCCAPT paper
# ---------------------------------------------------------------------------


def _tdc(rows: list[tuple[int, int]]) -> pd.DataFrame:
    """Helper: build a tdc-style dataframe from (event_group_id, channel) rows."""
    gid, ch = zip(*rows)
    return pd.DataFrame({
        "event_group_id": list(gid),
        "channel": list(ch),
    })


def test_classify_pulse_chunks_surface_concept_basic_complete():
    """A pulse with channels [0,1,2,3] is one complete (4-DLTS) chunk."""
    tdc = _tdc([(0, 0), (0, 1), (0, 2), (0, 3)])
    out = helper_auto_raw_analysis._classify_pulse_chunks(
        tdc, "event_group_id", "surface_concept"
    )
    assert out["frequency"].tolist() == [4]      # one pulse, length 4
    assert out["complete"].tolist() == [4]       # one complete chunk
    assert out["midtier"].tolist() == []
    assert out["partial"].tolist() == []


def test_classify_pulse_chunks_surface_concept_partial_x_only():
    """Channels [0,1,1] — one chunk with only the (0,1) pair → partial."""
    tdc = _tdc([(0, 0), (0, 1), (0, 1)])
    out = helper_auto_raw_analysis._classify_pulse_chunks(
        tdc, "event_group_id", "surface_concept"
    )
    assert out["complete"].tolist() == []
    assert out["partial"].tolist() == [3]


def test_classify_pulse_chunks_surface_concept_partial_y_only_no_legacy_bug():
    """Channels [2,3,3] — y-only partial. Legacy code had a copy-paste bug
    that silently dropped y-only partials from the orange bar; the new
    chunked classifier must count it correctly."""
    tdc = _tdc([(0, 2), (0, 3), (0, 3)])
    out = helper_auto_raw_analysis._classify_pulse_chunks(
        tdc, "event_group_id", "surface_concept"
    )
    assert out["complete"].tolist() == []
    assert out["partial"].tolist() == [3]   # ← would have been [] under the legacy bug


def test_classify_pulse_chunks_per_chunk_not_per_pulse():
    """The new chunked classifier must emit one entry **per chunk**, not per
    pulse — matching the legacy ``find_consecutive_sequences`` (and Figure 9A
    of the PyCCAPT paper). The previous per-pulse classifier capped the
    contribution of any length-N pulse to 1.

    A length-8 SC pulse with multiset {0,1,2,3,0,1,2,3} sorts to
    [0,0,1,1,2,2,3,3] and splits into two chunks-of-4: [0,0,1,1] (only the
    (0,1) pair fired → partial) and [2,2,3,3] (only the (2,3) pair → partial).
    So under sort-chunk semantics it contributes 2 entries to the *partial*
    bar at x=8 — not 1, and not in the complete bar.
    """
    tdc = _tdc([
        # pulse 0: clean 4-DLTS hit → one complete chunk
        (0, 0), (0, 1), (0, 2), (0, 3),
        # pulse 1: length-8 multi-hit
        (1, 0), (1, 1), (1, 2), (1, 3),
        (1, 0), (1, 1), (1, 2), (1, 3),
    ])
    out = helper_auto_raw_analysis._classify_pulse_chunks(
        tdc, "event_group_id", "surface_concept"
    )
    assert out["frequency"].tolist() == [4, 8]      # one entry per pulse
    # Per-chunk: pulse 0 → 1 complete; pulse 1 → 2 partial under sort-chunk.
    assert out["complete"].tolist() == [4]
    assert out["partial"].tolist() == [8, 8]
    # Sanity check: total chunks emitted = 3 (the per-pulse classifier would
    # have emitted at most 2, one per pulse).
    total_chunks = (
        len(out["complete"]) + len(out["midtier"]) + len(out["partial"])
    )
    assert total_chunks == 3


def test_classify_pulse_chunks_length_two_pair_counts_as_partial():
    """A length-2 SC pulse with channels [0, 1] *is* reconstructible in x,
    so the new classifier counts it as a 2-DLTS partial hit. (The legacy
    notebook's ``len(chs) > 2`` guard excluded these as noise; we
    deliberately diverge from that — the more permissive interpretation is
    the physics-correct one.)"""
    tdc = _tdc([(0, 0), (0, 1)])
    out = helper_auto_raw_analysis._classify_pulse_chunks(
        tdc, "event_group_id", "surface_concept"
    )
    assert out["frequency"].tolist() == [2]
    assert out["partial"].tolist()   == [2]
    assert out["complete"].tolist()  == []


def test_classify_pulse_chunks_length_two_y_pair_counts_as_partial():
    """Symmetric test: a length-2 SC pulse with channels [2, 3] is also a
    valid 2-DLTS partial (reconstructible in y)."""
    tdc = _tdc([(0, 2), (0, 3)])
    out = helper_auto_raw_analysis._classify_pulse_chunks(
        tdc, "event_group_id", "surface_concept"
    )
    assert out["partial"].tolist() == [2]


def test_classify_pulse_chunks_length_two_unrelated_channels_is_noise():
    """A length-2 chunk that does not have a full pair (e.g. [0, 2] or
    [0, 0]) still counts as noise — we only flip on chunks with at least
    one complete delay-line pair."""
    tdc = _tdc([(0, 0), (0, 2), (1, 0), (1, 0)])
    out = helper_auto_raw_analysis._classify_pulse_chunks(
        tdc, "event_group_id", "surface_concept"
    )
    # Pulse 0: [0, 2] — no pair complete. Pulse 1: [0, 0] — no pair complete.
    # Both contribute to frequency but not to partial/complete.
    assert out["frequency"].tolist() == [2, 2]
    assert out["partial"].tolist()   == []
    assert out["complete"].tolist()  == []


def test_classify_pulse_chunks_roentdek_three_categories():
    """RoentDek must distinguish 6 DLTS (all 3 pairs), 4 DLTS (2 of 3 pairs),
    and 2 DLTS (1 of 3 pairs)."""
    tdc = _tdc([
        # pulse 0: full hex hit  → complete
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
        # pulse 1: x + y pairs only, no z pair  → midtier (4 DLTS)
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 0), (1, 1),
        # pulse 2: only x pair  → partial (2 DLTS)
        (2, 0), (2, 1), (2, 1),
    ])
    out = helper_auto_raw_analysis._classify_pulse_chunks(
        tdc, "event_group_id", "roentdek"
    )
    assert out["complete"].tolist() == [6]
    assert out["midtier"].tolist()  == [6]
    assert out["partial"].tolist()  == [3]


def test_classify_pulse_chunks_single_delay_line():
    """A 1-DL system: a complete event is the (0,1) pair = 2 DLTS."""
    tdc = _tdc([(0, 0), (0, 1), (1, 0), (1, 0)])
    out = helper_auto_raw_analysis._classify_pulse_chunks(
        tdc, "event_group_id", "single_delay_line"
    )
    assert out["complete"].tolist() == [2]   # pulse 0 has the (0,1) pair
    # pulse 1 has [0,0] — no pair fully present, no partial
    assert out["partial"].tolist() == []


# ---------------------------------------------------------------------------
# pyccapt_raw_to_processed: mc / mc_uc preservation + computation
# ---------------------------------------------------------------------------


def test_pyccapt_raw_to_processed_preserves_existing_mc_and_mc_uc():
    """A frame that already has calibrated mc / uncalibrated mc_uc columns
    (e.g. from a partly-processed bundle) must keep them, NOT zero them."""
    raw = pd.DataFrame({
        "high_voltage (V)": [1500.0, 1500.0],
        "pulse_v (V)": [200.0, 200.0],
        "pulse_l (pJ)": [0.0, 0.0],
        "start_counter": [1, 2],
        "t (ns)": [400.0, 410.0],
        "x_det (cm)": [0.5, 0.6],
        "y_det (cm)": [-0.5, -0.6],
        "mc (Da)":    [27.0, 27.05],
        "mc_uc (Da)": [27.5, 27.55],
    })
    processed = data_tools.pyccapt_raw_to_processed(raw)
    assert processed["mc (Da)"].tolist()    == [27.0, 27.05]
    assert processed["mc_uc (Da)"].tolist() == [27.5, 27.55]


def test_pyccapt_raw_to_processed_computes_mc_uc_when_absent():
    """If the raw frame has no mc_uc column but has all the inputs (t, V,
    x_det, y_det), mc_uc is computed using tof2mc(t0=0, V_pulse=0, fpl=110).
    This is the formula the legacy raw-data notebook used for Fig. 6A."""
    raw = pd.DataFrame({
        "high_voltage (V)": [3000.0],
        "pulse_v (V)": [400.0],
        "pulse_l (pJ)": [0.0],
        "start_counter": [1],
        "t (ns)": [600.0],
        "x_det (cm)": [0.0],
        "y_det (cm)": [0.0],
    })
    processed = data_tools.pyccapt_raw_to_processed(raw)
    # Reference value from mc_tools.tof2mc directly — proves we're calling the
    # same formula instead of zeroing.
    from pyccapt.calibration.mc import mc_tools
    expected = mc_tools.tof2mc(
        t=np.array([600.0]),
        t0=0,
        V=np.array([3000.0]),
        xDet=np.array([0.0]),
        yDet=np.array([0.0]),
        flightPathLength=110,
        V_pulse=np.zeros(1),
        mode="voltage",
    )
    assert np.allclose(processed["mc_uc (Da)"].to_numpy(), expected)
    assert (processed["mc_uc (Da)"] != 0).all()
    # mc (calibrated) stays zero — there's no calibration here.
    assert (processed["mc (Da)"] == 0).all()


def test_run_analysis_accepts_peak_units_argument():
    """The top-level runner must accept ``peak_units='tof'`` and ``'mc'``
    without crashing on a tdc-less variables object."""
    variables = Variables()
    variables.data = pd.DataFrame({
        "mc (Da)": [27.0, 27.05],
        "t (ns)": [400.0, 410.0],
        "x_det (cm)": [0.0, 0.1],
        "y_det (cm)": [0.0, 0.1],
        "delta_p": [0, 1],
        "multi": [1, 1],
        "start_counter": [1, 2],
    })
    species = [{"label": "Al+", "mc_low": 26.78, "mc_up": 27.18, "color": "#ccc"}]

    captured: list[object] = []

    def _capture(obj):
        captured.append(obj)

    real_display = helper_auto_raw_analysis.display
    helper_auto_raw_analysis.display = _capture
    try:
        helper_auto_raw_analysis.run_analysis(variables, species, peak_units="tof")
        helper_auto_raw_analysis.run_analysis(variables, species, peak_units="mc")
        # Invalid unit normalizes to "tof" instead of raising.
        helper_auto_raw_analysis.run_analysis(variables, species, peak_units="bogus")
    finally:
        helper_auto_raw_analysis.display = real_display


def test_call_auto_raw_data_analysis_panel_has_save_dropdown_and_units_dropdown():
    """The redesigned panel must expose three Dropdowns (peak source, peak
    units, save plots) plus the six manual rows, summary, run button, and
    output. ``save plots`` must default to ``False``; ``peak units`` must
    default to ``'tof'``."""
    variables = Variables()
    variables.data = pd.DataFrame({
        "mc (Da)": [27.0],
        "t (ns)": [400.0],
        "x_det (cm)": [0.0],
        "y_det (cm)": [0.0],
        "delta_p": [0],
        "multi": [1],
        "start_counter": [1],
    })

    captured = {}
    real_display = helper_auto_raw_analysis.display
    helper_auto_raw_analysis.display = lambda obj: captured.setdefault("panel", obj)
    try:
        helper_auto_raw_analysis.call_auto_raw_data_analysis(variables)
    finally:
        helper_auto_raw_analysis.display = real_display

    panel = captured["panel"]
    assert isinstance(panel, widgets.VBox)

    dropdowns = [c for c in panel.children if isinstance(c, widgets.Dropdown)]
    # Four dropdowns: Peak source, Peak units, Save plots, Recovery.
    assert len(dropdowns) == 4
    descriptions = {d.description for d in dropdowns}
    assert {"Peak source:", "Peak units:", "Save plots:", "Recovery:"} <= descriptions

    by_desc = {d.description: d for d in dropdowns}
    assert by_desc["Save plots:"].value is False
    assert {value for _label, value in by_desc["Save plots:"].options} == {True, False}
    assert by_desc["Peak units:"].value == "tof"
    assert {value for _label, value in by_desc["Peak units:"].options} == {"tof", "mc"}
    # Recovery dropdown defaults to the combinatorial "exhaustive" mode
    # (slow but optimal); greedy is offered as a fast alternative. The
    # legacy per-chunk "fixed" mode has been removed entirely.
    assert by_desc["Recovery:"].value == "exhaustive"
    assert {value for _label, value in by_desc["Recovery:"].options} == {
        "greedy", "exhaustive",
    }


def test_plot_full_spectrum_renders_panels_for_available_signal_columns():
    """``plot_full_spectrum`` shows clean log-y histograms of TOF + M/C
    covering every event including noise (no peak shading). Panels for
    columns that don't exist or are all-zero are dropped silently."""
    import matplotlib

    matplotlib.use("Agg")

    # Calibrated bundle scenario: t_c (ns) populated → tof panel uses it,
    # mc (Da) populated → mc panel uses it. mc_uc absent. Two panels expected.
    dld = pd.DataFrame({
        "t (ns)":     np.linspace(50.0, 700.0, 200),
        "t_c (ns)":   np.linspace(60.0, 690.0, 200),
        "mc (Da)":    np.linspace(1.0,  60.0, 200),
        "mc_uc (Da)": np.zeros(200),
    })

    figs: list[object] = []
    real_display = helper_auto_raw_analysis.display
    helper_auto_raw_analysis.display = lambda obj: figs.append(obj)
    try:
        helper_auto_raw_analysis.plot_full_spectrum(dld)
    finally:
        helper_auto_raw_analysis.display = real_display

    # Two panels (tof_c + mc), no peak shading anywhere.
    assert len(figs) == 1
    fig = figs[0]
    axes = fig.axes
    assert len(axes) == 2
    titles = [ax.get_title() for ax in axes]
    assert any("Full time-of-flight spectrum" in t for t in titles)
    assert any("Full mass spectrum" in t for t in titles)
    # No axvspan was added (peak shading is for the per-peak plots, not here).
    for ax in axes:
        assert not ax.collections, "full-spectrum plot must not draw peak overlays"


def test_plot_full_spectrum_handles_pure_raw_dld_with_only_t_and_mc_uc():
    """Pure raw acquisition: t_c and mc are zero, but t and mc_uc are
    populated. The fallback selectors must still yield a 2-panel figure
    (using ``t (ns)`` and ``mc_uc (Da)``)."""
    import matplotlib

    matplotlib.use("Agg")

    dld = pd.DataFrame({
        "t (ns)":     np.linspace(50.0, 700.0, 200),
        "t_c (ns)":   np.zeros(200),                 # never calibrated
        "mc (Da)":    np.zeros(200),                 # never calibrated
        "mc_uc (Da)": np.linspace(1.0, 60.0, 200),   # raw mc only
    })

    figs: list[object] = []
    real_display = helper_auto_raw_analysis.display
    helper_auto_raw_analysis.display = lambda obj: figs.append(obj)
    try:
        helper_auto_raw_analysis.plot_full_spectrum(dld)
    finally:
        helper_auto_raw_analysis.display = real_display

    assert len(figs) == 1
    titles = [ax.get_title() for ax in figs[0].axes]
    # Falls back to t (ns) for TOF and mc_uc (Da) for MC.
    assert any("(t (ns))" in t for t in titles)
    assert any("(mc_uc (Da))" in t for t in titles)


def test_call_auto_raw_data_analysis_run_button_disables_during_processing():
    """While the analysis is running, the run button + all input controls
    must be disabled and the button label must indicate "Processing…".
    After the run finishes (or raises), all controls must be re-enabled."""
    variables = Variables()
    variables.data = pd.DataFrame({
        "mc (Da)":    [27.0],
        "mc_uc (Da)": [27.0],
        "t (ns)":     [400.0],
        "t_c (ns)":   [0.0],
        "x_det (cm)": [0.0],
        "y_det (cm)": [0.0],
        "delta_p":    [0],
        "multi":      [1],
        "start_counter": [1],
    })

    captured: dict = {}
    real_display = helper_auto_raw_analysis.display
    helper_auto_raw_analysis.display = lambda obj: captured.setdefault("panel", obj)
    try:
        helper_auto_raw_analysis.call_auto_raw_data_analysis(variables)
    finally:
        helper_auto_raw_analysis.display = real_display

    panel = captured["panel"]
    dropdowns = [c for c in panel.children if isinstance(c, widgets.Dropdown)]
    by_desc = {d.description: d for d in dropdowns}

    # Locate the run button and the manual rows.
    buttons = [c for c in panel.children if isinstance(c, widgets.Button)]
    assert len(buttons) == 1
    run_button = buttons[0]

    # All controls start enabled.
    for d in dropdowns:
        assert not d.disabled
    assert not run_button.disabled
    assert run_button.description == "Run analysis"

    # Hijack the run handler to inspect the busy state mid-flight.
    seen_busy: dict = {}

    def _real_run_analysis(*_args, **_kwargs):
        seen_busy["dropdowns_disabled"] = [d.disabled for d in dropdowns]
        seen_busy["run_button_disabled"] = run_button.disabled
        seen_busy["run_button_label"] = run_button.description

    real_run = helper_auto_raw_analysis.run_analysis
    helper_auto_raw_analysis.run_analysis = _real_run_analysis
    try:
        # Switch to manual mode + supply at least one peak so _on_run actually
        # calls run_analysis (otherwise it would short-circuit on empty-peaks).
        by_desc["Peak source:"].value = "manual"
        # Trigger the run.
        run_button.click()
    finally:
        helper_auto_raw_analysis.run_analysis = real_run

    # During processing the controls were all disabled and the button label
    # advertised the busy state.
    assert all(seen_busy["dropdowns_disabled"]), seen_busy["dropdowns_disabled"]
    assert seen_busy["run_button_disabled"] is True
    assert seen_busy["run_button_label"] == "Processing…"

    # After processing the controls are re-enabled and the label is restored.
    for d in dropdowns:
        assert not d.disabled
    assert not run_button.disabled
    assert run_button.description == "Run analysis"


def test_call_auto_raw_data_analysis_runs_with_no_peaks_when_manual_rows_empty():
    """When the user clicks Run analysis with no peak windows typed, the
    runner must still proceed (skipping the per-peak sections) instead of
    short-circuiting with a hard error."""
    variables = Variables()
    variables.data = pd.DataFrame({
        "mc (Da)":    [27.0],
        "mc_uc (Da)": [27.0],
        "t (ns)":     [400.0],
        "t_c (ns)":   [0.0],
        "x_det (cm)": [0.0],
        "y_det (cm)": [0.0],
        "delta_p":    [0],
        "multi":      [1],
        "start_counter": [1],
    })

    captured: dict = {}
    real_display = helper_auto_raw_analysis.display
    helper_auto_raw_analysis.display = lambda obj: captured.setdefault("panel", obj)
    try:
        helper_auto_raw_analysis.call_auto_raw_data_analysis(variables)
    finally:
        helper_auto_raw_analysis.display = real_display

    panel = captured["panel"]
    dropdowns = [c for c in panel.children if isinstance(c, widgets.Dropdown)]
    by_desc = {d.description: d for d in dropdowns}
    by_desc["Peak source:"].value = "manual"
    buttons = [c for c in panel.children if isinstance(c, widgets.Button)]
    run_button = buttons[0]

    # Capture: was run_analysis called? With what species?
    received: dict = {}

    def _capture_run(_variables, species, **_kwargs):
        received["species"] = list(species)

    real_run = helper_auto_raw_analysis.run_analysis
    helper_auto_raw_analysis.run_analysis = _capture_run
    try:
        run_button.click()       # manual rows are all 0 → empty species
    finally:
        helper_auto_raw_analysis.run_analysis = real_run

    assert "species" in received, (
        "_on_run must call run_analysis even when no peaks are supplied — the "
        "global sections (DLTS-per-pulse, full TOF/MC, FDM, multi-hit) "
        "should still render."
    )
    assert received["species"] == []


def test_call_signal_preview_panel_lists_only_available_targets():
    """The preview panel offers a Target dropdown whose options are limited to
    the columns that actually have non-zero data on ``variables.data``. So a
    file with only ``t (ns)`` + ``mc (Da)`` populated must show those two
    targets and hide ``tof_c`` / ``mc_uc`` entirely."""
    variables = Variables()
    variables.data = pd.DataFrame({
        "t (ns)":     [400.0, 410.0, 420.0],   # available → "tof"
        "t_c (ns)":   [0.0, 0.0, 0.0],         # all-zero → hidden
        "mc (Da)":    [27.0, 27.05, 27.1],     # available → "mc"
        # mc_uc (Da) absent entirely → hidden
        "high_voltage (V)": [1500.0, 1500.0, 1500.0],
    })

    captured = {}
    real_display = helper_auto_raw_analysis.display
    helper_auto_raw_analysis.display = lambda obj: captured.setdefault("panel", obj)
    try:
        helper_auto_raw_analysis.call_signal_preview(variables)
    finally:
        helper_auto_raw_analysis.display = real_display

    panel = captured["panel"]
    assert isinstance(panel, widgets.VBox)

    dropdowns = []

    def _walk(node):
        if isinstance(node, widgets.Dropdown):
            dropdowns.append(node)
        children = getattr(node, "children", None)
        if children:
            for child in children:
                _walk(child)

    _walk(panel)
    target_dropdowns = [d for d in dropdowns if {value for _label, value in d.options}
                        & {"tof", "tof_c", "mc", "mc_uc"}]
    assert len(target_dropdowns) == 1
    target_values = {value for _label, value in target_dropdowns[0].options}
    assert target_values == {"tof", "mc"}        # the two with data
    assert "tof_c" not in target_values
    assert "mc_uc" not in target_values


def test_call_signal_preview_peak_find_propagates_print_and_overlay_flags():
    """When the user enables ``Peak find`` in the preview panel, the call
    into :func:`mc_plot.hist_plot` must set ``peaks_find=True``,
    ``peaks_find_plot=True`` (peaks drawn on the histogram), and
    ``print_info=True`` (peak locations + left/right window edges + MRP
    printed beneath the figure). When Peak find is False, all three must
    be False so we don't pay for unwanted peak-finding work."""
    variables = Variables()
    variables.data = pd.DataFrame({
        "t (ns)":  [400.0, 410.0, 420.0],
        "mc (Da)": [27.0, 27.05, 27.1],
    })

    captured = {}
    real_display = helper_auto_raw_analysis.display
    helper_auto_raw_analysis.display = lambda obj: captured.setdefault("panel", obj)
    try:
        helper_auto_raw_analysis.call_signal_preview(variables)
    finally:
        helper_auto_raw_analysis.display = real_display

    panel = captured["panel"]
    # Locate buttons and the Peak-find dropdown by walking the panel.
    plot_button = None
    peaks_find = None

    def _walk(node):
        nonlocal plot_button, peaks_find
        if isinstance(node, widgets.Button) and node.description == "Plot":
            plot_button = node
        if isinstance(node, widgets.Dropdown):
            opts = {value for _label, value in node.options}
            if opts == {True, False} and peaks_find is None:
                # This may match multiple True/False dropdowns — assign the
                # first one and keep walking; rebind below using the
                # surrounding label for confidence.
                peaks_find = node
        for child in getattr(node, "children", None) or ():
            _walk(child)

    _walk(panel)
    # Disambiguate Peak find: it's the dropdown sandwiched between a
    # Label("Peak find:") and the explanation HTML.
    for child in panel.children:
        if isinstance(child, widgets.HBox) and len(child.children) == 2:
            label, dropdown = child.children
            if (
                isinstance(label, widgets.Label)
                and label.value.strip().rstrip(":") == "Peak find"
                and isinstance(dropdown, widgets.Dropdown)
            ):
                peaks_find = dropdown
                break
    assert peaks_find is not None and plot_button is not None

    captured_calls: list[dict] = []

    def _fake_hist_plot(_variables, _bin_size, **kwargs):
        captured_calls.append(kwargs)

    from pyccapt.calibration.core import mc_plot
    real_hist = mc_plot.hist_plot
    mc_plot.hist_plot = _fake_hist_plot
    try:
        # Peak find = True → all three flags propagate as True.
        peaks_find.value = True
        plot_button.click()
        # Peak find = False → all three are False.
        peaks_find.value = False
        plot_button.click()
    finally:
        mc_plot.hist_plot = real_hist

    assert len(captured_calls) == 2

    on_call, off_call = captured_calls
    assert on_call["peaks_find"]      is True
    assert on_call["peaks_find_plot"] is True
    assert on_call["print_info"]      is True

    assert off_call["peaks_find"]      is False
    assert off_call["peaks_find_plot"] is False
    assert off_call["print_info"]      is False


def test_close_after_skips_close_on_interactive_backends():
    """``_close_after`` must keep figures alive on interactive backends so
    the user can still zoom / pan / save them. Static backends (inline /
    Agg) close as before to free memory."""
    import matplotlib
    import matplotlib.pyplot as plt

    real_get_backend = plt.get_backend

    def _make_fig():
        return plt.figure()

    # Interactive backend (mocked) → figure stays open.
    fig_interactive = _make_fig()
    plt.get_backend = lambda: "module://ipympl.backend_nbagg"
    try:
        helper_auto_raw_analysis._close_after(fig_interactive)
        # Figure should still be in pyplot's number list (alive).
        assert plt.fignum_exists(fig_interactive.number), (
            "interactive backend: _close_after must NOT close the figure "
            "(otherwise the canvas freezes and the user loses zoom)"
        )
    finally:
        plt.get_backend = real_get_backend
        plt.close(fig_interactive)

    # Static backend → figure is closed.
    fig_static = _make_fig()
    plt.get_backend = lambda: "module://matplotlib_inline.backend_inline"
    try:
        helper_auto_raw_analysis._close_after(fig_static)
        assert not plt.fignum_exists(fig_static.number), (
            "static backend: _close_after must close the figure to free memory"
        )
    finally:
        plt.get_backend = real_get_backend
        # Already closed in the static branch; calling close again is a no-op.
        plt.close(fig_static)


def test_call_signal_preview_handles_empty_dataframe():
    """If every candidate column is missing or all-zero, the helper should
    print a Markdown message instead of trying to render an empty panel."""
    variables = Variables()
    variables.data = pd.DataFrame({
        "t (ns)":   [0.0, 0.0],
        "t_c (ns)": [0.0, 0.0],
        # no mc / mc_uc
    })

    md_messages: list[str] = []
    real_display = helper_auto_raw_analysis.display

    def _capture(obj):
        text = getattr(obj, "data", None)
        if isinstance(text, str):
            md_messages.append(text)

    helper_auto_raw_analysis.display = _capture
    try:
        helper_auto_raw_analysis.call_signal_preview(variables)
    finally:
        helper_auto_raw_analysis.display = real_display

    assert any("cannot render the preview" in m for m in md_messages)


def test_pyccapt_raw_to_processed_load_calibrated_h5_pure_raw_now_has_mc_uc(tmp_path: Path):
    """End-to-end: load a pure-raw acquisition file via load_calibrated_h5
    (which falls back to fetch_dataset_with_tdc + pyccapt_raw_to_processed)
    and verify the resulting frame has a real mc_uc column instead of zeros.

    This is the regression that broke the M/C histogram on raw files — it
    must stay green."""
    import h5py

    dld_sc = np.array([1, 2, 3], dtype=np.int64)
    tdc_sc = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int64)
    h5_path = tmp_path / "raw_only_mc_uc.h5"
    n_dld = len(dld_sc)
    n_tdc = len(tdc_sc)
    with h5py.File(h5_path, "w") as hdf:
        dld_grp = hdf.create_group("dld")
        dld_grp.create_dataset("high_voltage", data=np.full((n_dld, 1), 3000.0))
        dld_grp.create_dataset("voltage_pulse", data=np.full((n_dld, 1), 400.0))
        dld_grp.create_dataset("laser_intensity", data=np.zeros((n_dld, 1)))
        dld_grp.create_dataset("start_counter", data=dld_sc.reshape(-1, 1))
        dld_grp.create_dataset("t", data=np.array([[400.0], [500.0], [600.0]]))
        dld_grp.create_dataset("x", data=np.full((n_dld, 1), 0.5))
        dld_grp.create_dataset("y", data=np.full((n_dld, 1), -0.5))

        tdc_grp = hdf.create_group("tdc")
        tdc_grp.create_dataset("channel", data=(np.arange(n_tdc) % 4).reshape(-1, 1).astype(np.int64))
        tdc_grp.create_dataset("start_counter", data=tdc_sc.reshape(-1, 1))
        tdc_grp.create_dataset("high_voltage", data=np.full((n_tdc, 1), 3000.0))
        tdc_grp.create_dataset("voltage_pulse", data=np.full((n_tdc, 1), 400.0))
        tdc_grp.create_dataset("laser_pulse", data=np.zeros((n_tdc, 1)))
        tdc_grp.create_dataset("time_data", data=np.linspace(40.0, 80.0, n_tdc).reshape(-1, 1))

    variables = Variables()
    loaded_dld, _, _ = helper_data_loader.load_calibrated_h5(str(h5_path), variables)

    assert "mc_uc (Da)" in loaded_dld.columns
    assert (loaded_dld["mc_uc (Da)"] != 0).all(), (
        "mc_uc must be computed from raw inputs, not zero — otherwise the M/C "
        "histogram is silently dropped on pure-raw acquisition files."
    )
