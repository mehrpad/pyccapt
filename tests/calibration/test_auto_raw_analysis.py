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


def test_detect_detector_kind_unknown_for_empty_or_missing():
    assert helper_auto_raw_analysis.detect_detector_kind(None) == "unknown"
    assert helper_auto_raw_analysis.detect_detector_kind(pd.DataFrame()) == "unknown"


def test_expected_dlts_full():
    assert helper_auto_raw_analysis.expected_dlts_full("surface_concept") == 4
    assert helper_auto_raw_analysis.expected_dlts_full("roentdek") == 6
    assert helper_auto_raw_analysis.expected_dlts_full("unknown") == 0


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
    # First child is the peak-source dropdown.
    dropdown = panel.children[0]
    assert isinstance(dropdown, widgets.Dropdown)
    assert {value for _label, value in dropdown.options} == {"manual", "range"}

    save_plots = panel.children[2]
    assert isinstance(save_plots, widgets.Checkbox)
    assert save_plots.value is False

    # Manual rows are nested after the summary and save checkbox.
    manual_grid = panel.children[3]
    assert isinstance(manual_grid, widgets.VBox)
    assert len(manual_grid.children) == 6   # exactly six peak rows

    def _all_disabled():
        return all(
            child.children[0].disabled
            and child.children[1].disabled
            and child.children[2].disabled
            for child in manual_grid.children
        )

    # No range data was loaded, so the dropdown defaults to "manual" and rows are enabled.
    assert dropdown.value == "manual"
    assert not _all_disabled()

    # Toggling to "range" must disable the rows.
    dropdown.value = "range"
    assert _all_disabled()

    # Toggling back to "manual" re-enables them.
    dropdown.value = "manual"
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
