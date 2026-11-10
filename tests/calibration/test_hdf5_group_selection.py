"""Hit-only import never materializes unrelated detector-channel events."""
import h5py
import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.data_tools import data_loadcrop, data_tools


def _block_other_group_reads(monkeypatch, group):
    original = h5py.Dataset.__array__
    reads = []

    def guarded(dataset, *args, **kwargs):
        if not dataset.name.startswith(f"/{group}/"):
            raise AssertionError(f"Unrequested array materialized: {dataset.name}")
        reads.append(dataset.name)
        return original(dataset, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__array__", guarded)
    return reads


@pytest.mark.parametrize("shape", ["vector", "column"])
@pytest.mark.parametrize("pulse_alias", ["pulse", "voltage_pulse", "pulse_voltage"])
def test_dld_only_read_preserves_alias_shape_units_and_integer_precision(tmp_path, monkeypatch, shape, pulse_alias):
    path = tmp_path / "hits_and_large_channels.h5"
    counters = np.array([2**60+7, 2**60+8, 2**60+11], dtype=np.uint64)
    source = {"high_voltage": np.array([500., 900., 1200.]), pulse_alias: np.array([328., 329., 330.]),
              "laser_intensity": np.array([0., 1., 2.]), "start_counter": counters,
              "t": np.array([100., 200., 300.]), "x": np.array([1200., 1225., 1300.]),
              "y": np.array([1225., 1200., 1250.])}
    with h5py.File(path, "w") as handle:
        dld = handle.create_group("dld")
        for key, values in source.items():
            dld.create_dataset(key, data=values[:, None] if shape == "column" else values)
        # An unallocated large dataset costs virtually no disk space. Reading
        # it would allocate 200 MB, but the guard fails before any such read.
        handle.create_dataset("tdc/time_data", shape=(25_105_070,), dtype="u8", chunks=(1024,))
        handle.create_dataset("apt/time_counter", data=np.arange(5))
    reads = _block_other_group_reads(monkeypatch, "dld")
    actual = data_tools.load_data(path, "pyccapt", mode="raw")
    expected = pd.DataFrame({"high_voltage (V)": source["high_voltage"], "pulse_v (V)": source[pulse_alias],
        "pulse_l (pJ)": source["laser_intensity"], "start_counter": counters,
        "t (ns)": source["t"], "x_det (cm)": source["x"], "y_det (cm)": source["y"]})
    if pulse_alias == "pulse_voltage":
        # Independent legacy bin-to-ns / firmware-window bin-to-cm contract
        # (centre pixel 1201 / 1190, pix_size_x / pix_size_y of tdc_gpx3.ini).
        expected["t (ns)"] *= 27.432 / 4000
        expected["x_det (cm)"] = (expected["x_det (cm)"] + .5 - 1201.) * .033306 * .1
        expected["y_det (cm)"] = (expected["y_det (cm)"] + .5 - 1190.) * .033613 * .1
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual["start_counter"].dtype == np.uint64
    assert set(reads) == {f"/dld/{name}" for name in source}
    assert not actual.attrs.get("pyccapt_unavailable_columns")


@pytest.mark.parametrize("mode", ["tdc_sc", "tdc_ro"])
def test_tdc_only_read_avoids_dld_and_keeps_missing_laser_semantics(tmp_path, monkeypatch, mode):
    path = tmp_path / "channels_and_hits.h5"
    big = np.array([2**60+3, 2**60+9], dtype=np.uint64)
    with h5py.File(path, "w") as handle:
        for key, values in {"channel": np.array([1, 7], dtype=np.uint32), "start_counter": big,
                "high_voltage": [500., 600.], "pulse": [328., 329.], "time_data": big+4}.items():
            handle.create_dataset(f"tdc/{key}", data=np.asarray(values)[:, None])
        handle.create_dataset("dld/t", data=np.arange(10.))
    reads = _block_other_group_reads(monkeypatch, "tdc")
    actual = data_loadcrop.fetch_dataset_from_dld_grp(path, extract_mode=mode)
    expected = pd.DataFrame({"channel": np.array([1, 7], dtype=np.uint32), "start_counter": big,
        "high_voltage (V)": [500., 600.], "pulse_v (V)": [328., 329.], "pulse_l (pJ)": [0., 0.], "time_data": big+4})
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.attrs["pyccapt_unavailable_columns"] == ["pulse_l (pJ)"]
    assert len(reads) == 5


def test_selective_dld_preserves_missing_optional_columns(tmp_path, monkeypatch):
    path = tmp_path / "minimal_hits.h5"
    with h5py.File(path, "w") as handle:
        for key, values in {"high_voltage": [500., 600.], "t": [100., 200.], "x": [0., 1.], "y": [0., -1.]}.items():
            handle.create_dataset(f"dld/{key}", data=values)
        handle.create_dataset("tdc/time_data", data=np.arange(10))
    _block_other_group_reads(monkeypatch, "dld")
    actual = data_loadcrop.fetch_dataset_from_dld_grp(path)
    assert actual.attrs["pyccapt_unavailable_columns"] == ["pulse_v (V)", "pulse_l (pJ)", "start_counter"]
    for column in actual.attrs["pyccapt_unavailable_columns"]:
        np.testing.assert_array_equal(actual[column], [0, 0])
    assert actual["start_counter"].dtype == np.uint64


def test_public_eager_default_and_lazy_reads_retain_all_groups(tmp_path):
    path = tmp_path / "all_groups.h5"
    with h5py.File(path, "w") as handle:
        for group in ("dld", "tdc", "apt"):
            handle.create_dataset(f"{group}/t", data=np.arange(3., dtype=float))
    assert set(data_tools.read_hdf5(path)) == {"dld/t", "tdc/t", "apt/t"}
    assert set(data_tools.read_hdf5(path, groups=("dld",))) == {"dld/t"}
    with data_tools.read_hdf5(path, lazy=True) as table:
        assert set(table.columns) == {"dld/t", "tdc/t"}
        np.testing.assert_array_equal(table["tdc/t"][:], np.arange(3.))
    with pytest.raises(ValueError, match="eager"):
        data_tools.read_hdf5(path, groups=("dld",), lazy=True)


@pytest.mark.parametrize("unit, factor", [("mm", 0.1), ("cm", 1.0), (None, 1.0)])
def test_dld_xy_unit_attribute_is_converted_to_cm(tmp_path, unit, factor):
    # RoentDek runs 283/284 (2026-11) wrote dld/x, dld/y in mm with unit = 'mm'.
    path = tmp_path / "run.h5"
    x = np.array([-39.4, 12.5, 39.9])
    y = np.array([-40.8, -3.0, 39.9])
    with h5py.File(path, "w") as handle:
        handle["dld/high_voltage"] = np.full(3, 5000.0)
        handle["dld/t"] = np.array([100.0, 200.0, 300.0])
        for name, values in (("x", x), ("y", y)):
            dataset = handle.create_dataset(f"dld/{name}", data=values)
            if unit is not None:
                dataset.attrs["unit"] = unit
    frame = data_loadcrop.fetch_dataset_from_dld_grp(str(path), "dld")
    np.testing.assert_allclose(frame["x_det (cm)"], x * factor)
    np.testing.assert_allclose(frame["y_det (cm)"], y * factor)
    with data_loadcrop.fetch_dataset_from_dld_grp(str(path), "dld", lazy=True) as table:
        np.testing.assert_allclose(table["x_det (cm)"][:], x * factor)
        np.testing.assert_allclose(np.asarray(table["y_det (cm)"]), y * factor)


def test_crop_figures_are_saved_without_the_interactive_selectors(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    from types import SimpleNamespace

    seen = []

    def record(fig, **kwargs):
        selectors = data_loadcrop.selectors_data.toggle_selector
        current = selectors.RS if kwargs["stem"] == "history" else selectors.ES
        seen.append((kwargs["stem"], any(artist.get_visible() for artist in current.artists), current.useblit))

    monkeypatch.setattr(data_loadcrop, "save_figure", record)
    monkeypatch.setattr(data_loadcrop.plt, "show", lambda *a, **k: None)
    rng = np.random.default_rng(0)
    n = 2000
    frame = pd.DataFrame({"t (ns)": rng.uniform(100, 2000, n), "high_voltage (V)": np.linspace(1000, 5000, n),
                          "pulse_l (pJ)": np.full(n, 2.0), "pulse_v (V)": np.zeros(n),
                          "x_det (cm)": rng.uniform(-3, 3, n), "y_det (cm)": rng.uniform(-3, 3, n)})
    variables = SimpleNamespace(result_path=str(tmp_path), selected_x1=100, selected_x2=900,
                                selected_x_fdm=0.0, selected_y_fdm=0.0, roi_fdm=2.0)
    data_loadcrop.plot_crop_experiment_history(frame, variables, 0, draw_rect=True, data_crop=True,
                                               pulse_plot=True, pulse_mode="laser", save=True, figname="history")
    data_loadcrop.plot_crop_fdm(frame["x_det (cm)"].to_numpy(), frame["y_det (cm)"].to_numpy(), variables=variables,
                                data_crop=True, draw_circle=True, save=True, figname="fdm")
    # hidden (and not blitting) while saved ...
    assert seen == [("history", False, False), ("fdm", False, False)]
    # ... and back for further cropping afterwards
    selectors = data_loadcrop.selectors_data.toggle_selector
    assert selectors.RS.artists[0].get_visible() and selectors.ES.artists[0].get_visible()
    assert selectors.RS.useblit and selectors.ES.useblit


def test_event_index_is_the_start_counter_when_dld_has_none(tmp_path):
    # RoentDek runs 283/284 (2026-11) store the TDC group of each ion in
    # dld/event_index and have no dld/start_counter; zeros made every ion a
    # multi-hit (100 % "multihit events per pulse").
    path = tmp_path / "run.h5"
    with h5py.File(path, "w") as handle:
        handle["dld/high_voltage"] = np.full(4, 5000.0)
        handle["dld/t"] = np.array([100.0, 200.0, 300.0, 400.0])
        handle["dld/x"] = np.zeros(4)
        handle["dld/y"] = np.zeros(4)
        handle["dld/event_index"] = np.array([0, 2, 2, 5], dtype=np.int64)
    frame = data_loadcrop.fetch_dataset_from_dld_grp(str(path), "dld")
    np.testing.assert_array_equal(frame["start_counter"], [0, 2, 2, 5])
    assert "start_counter" not in frame.attrs.get("pyccapt_unavailable_columns", [])
    with data_loadcrop.fetch_dataset_from_dld_grp(str(path), "dld", lazy=True) as table:
        np.testing.assert_array_equal(table["start_counter"][:], [0, 2, 2, 5])
