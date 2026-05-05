from pathlib import Path

import pandas as pd
import pytest

from pyccapt.calibration.core.share_variables import Variables
from pyccapt.calibration.data_tools import ato_tools, data_tools
from pyccapt.calibration.leap_tools import ccapt_tools


def test_store_df_to_hdf_supports_modern_argument_order(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out_file = tmp_path / "modern.h5"

    data_tools.store_df_to_hdf(df, "df", out_file)

    loaded = pd.read_hdf(out_file, "df")
    assert loaded.equals(df)


def test_store_df_to_hdf_supports_legacy_argument_order(tmp_path: Path):
    df = pd.DataFrame({"a": [10], "b": [20]})
    out_file = tmp_path / "legacy.h5"

    data_tools.store_df_to_hdf(out_file, df, "df")

    loaded = pd.read_hdf(out_file, "df")
    assert loaded.equals(df)


def test_load_data_rejects_unknown_mode_for_pyccapt():
    with pytest.raises(ValueError):
        data_tools.load_data("dummy.h5", data_type="pyccapt", mode="unknown")


def test_load_data_processed_handles_missing_pulse_columns(tmp_path: Path):
    dataset_path = tmp_path / "missing_pulse.h5"
    df = pd.DataFrame(
        {
            "high_voltage (V)": [1000.0, 1001.0],
            "t (ns)": [100.0, 101.0],
            "x_det (cm)": [0.1, 0.2],
            "y_det (cm)": [0.2, 0.3],
        }
    )
    df.to_hdf(dataset_path, key="df", mode="w")

    with pytest.warns(RuntimeWarning, match="does not have 'pulse'"):
        loaded = data_tools.load_data(str(dataset_path), data_type="pyccapt", mode="processed")

    assert "pulse_v (V)" in loaded.columns
    assert "pulse_l (pJ)" in loaded.columns
    assert loaded["pulse_v (V)"].tolist() == pytest.approx([0.0, 0.0])
    assert loaded["pulse_l (pJ)"].tolist() == pytest.approx([0.0, 0.0])


def test_load_data_processed_maps_legacy_pulse_column(tmp_path: Path):
    dataset_path = tmp_path / "legacy_pulse.h5"
    df = pd.DataFrame(
        {
            "high_voltage (V)": [1000.0, 1001.0],
            "pulse": [12.0, 13.0],
            "t (ns)": [100.0, 101.0],
            "x_det (cm)": [0.1, 0.2],
            "y_det (cm)": [0.2, 0.3],
        }
    )
    df.to_hdf(dataset_path, key="df", mode="w")

    loaded = data_tools.load_data(str(dataset_path), data_type="pyccapt", mode="processed")

    assert "pulse" not in loaded.columns
    assert loaded["pulse_v (V)"].tolist() == pytest.approx([12.0, 13.0])
    assert loaded["pulse_l (pJ)"].tolist() == pytest.approx([0.0, 0.0])


def test_load_data_raw_with_tdc_falls_back_to_roentdek_extract_mode(monkeypatch):
    expected = (pd.DataFrame({"start_counter": [1]}), pd.DataFrame({"start_counter": [1], "channel": [0]}))
    calls = []

    def fake_fetch_dataset_with_tdc(path, tdc_extract_mode="tdc_sc"):
        calls.append(tdc_extract_mode)
        if tdc_extract_mode == "tdc_sc":
            raise ValueError("surface extractor mismatch")
        return expected

    monkeypatch.setattr("pyccapt.calibration.data_tools.data_loadcrop.fetch_dataset_with_tdc", fake_fetch_dataset_with_tdc)

    loaded = data_tools.load_data("dummy.h5", data_type="pyccapt", mode="raw", load_tdc=True)

    assert loaded == expected
    assert calls == ["tdc_sc", "tdc_ro"]


def test_read_range_h5_backfills_name_from_ion(tmp_path: Path):
    range_path = tmp_path / "legacy_range.h5"
    legacy_range = pd.DataFrame(
        {
            "ion": ["Ni", "Ni3Al"],
            "mc_low": [57.8, 175.8],
            "mc_up": [58.2, 176.4],
            "color": ["#00AAFF", "#AA5500"],
        }
    )
    legacy_range.to_hdf(range_path, key="df", mode="w")

    normalized = data_tools.read_range(range_path)

    assert "name" in normalized.columns
    assert normalized["name"].tolist() == ["Ni", "Ni3Al"]


def test_read_range_h5_name_is_neutral_when_ion_has_charge(tmp_path: Path):
    range_path = tmp_path / "legacy_range_charged.h5"
    legacy_range = pd.DataFrame(
        {
            "ion": ["$Fe^{2+}$", "$Fe^{+}$", "$Co^{2+}$"],
            "mc_low": [55.64, 55.74, 58.73],
            "mc_up": [56.04, 56.14, 59.13],
            "color": ["#DF84F0", "#B377D8", "#2AF43D"],
        }
    )
    legacy_range.to_hdf(range_path, key="df", mode="w")

    normalized = data_tools.read_range(range_path)

    assert normalized["name"].tolist() == ["Fe", "Fe", "Co"]


def test_read_range_h5_replaces_charged_name_when_element_available(tmp_path: Path):
    range_path = tmp_path / "legacy_range_name_charged.h5"
    legacy_range = pd.DataFrame(
        {
            "name": ["$Fe^{2+}$", "$Fe^{+}$"],
            "ion": ["$Fe^{2+}$", "$Fe^{+}$"],
            "element": [["Fe"], ["Fe"]],
            "complex": [[1], [1]],
            "mc_low": [55.64, 55.74],
            "mc_up": [56.04, 56.14],
            "color": ["#DF84F0", "#B377D8"],
        }
    )
    legacy_range.to_hdf(range_path, key="df", mode="w")

    normalized = data_tools.read_range(range_path)

    assert normalized["name"].tolist() == ["Fe", "Fe"]


def test_extract_data_accepts_legacy_high_voltage_alias():
    variables = Variables()
    data = pd.DataFrame(
        {
            "Voltage": [5000.0, 5100.0],
            "pulse_v (V)": [10.0, 10.5],
            "pulse_l (pJ)": [0.0, 0.0],
            "t (ns)": [100.0, 120.0],
            "x_det (cm)": [0.1, 0.2],
            "y_det (cm)": [0.3, 0.4],
        }
    )

    data_tools.extract_data(data, variables, flightPathLength_d=100.0, max_mc=120.0)

    assert variables.dld_high_voltage.tolist() == pytest.approx([5000.0, 5100.0])


def test_extract_data_falls_back_when_high_voltage_missing():
    variables = Variables()
    data = pd.DataFrame(
        {
            "pulse_v (V)": [10.0, 10.5],
            "pulse_l (pJ)": [0.0, 0.0],
            "t (ns)": [100.0, 120.0],
            "x_det (cm)": [0.1, 0.2],
            "y_det (cm)": [0.3, 0.4],
        }
    )

    with pytest.warns(RuntimeWarning, match="does not have a high-voltage column"):
        data_tools.extract_data(data, variables, flightPathLength_d=100.0, max_mc=120.0)

    assert variables.dld_high_voltage.tolist() == pytest.approx([0.0, 0.0])


def test_extract_data_accepts_legacy_tof_and_detector_mm_columns():
    variables = Variables()
    data = pd.DataFrame(
        {
            "high_voltage (V)": [5000.0, 5100.0],
            "pulse_v (V)": [10.0, 10.5],
            "TOF (ns)": [100.0, 120.0],
            "det_x (mm)": [1.0, 2.0],
            "det_y (mm)": [3.0, 4.0],
        }
    )

    with pytest.warns(RuntimeWarning, match="Using legacy TOF column"):
        data_tools.extract_data(data, variables, flightPathLength_d=100.0, max_mc=120.0)

    assert variables.dld_t.tolist() == pytest.approx([100.0, 120.0])
    assert variables.dld_t_calib.tolist() == pytest.approx([100.0, 120.0])
    assert variables.dld_x_det.tolist() == pytest.approx([0.1, 0.2])
    assert variables.dld_y_det.tolist() == pytest.approx([0.3, 0.4])


def test_extract_data_raises_when_tof_columns_missing():
    variables = Variables()
    data = pd.DataFrame(
        {
            "high_voltage (V)": [5000.0, 5100.0],
            "pulse_v (V)": [10.0, 10.5],
            "x_det (cm)": [0.1, 0.2],
            "y_det (cm)": [0.3, 0.4],
        }
    )

    with pytest.raises(KeyError, match="No TOF column found"):
        data_tools.extract_data(data, variables, flightPathLength_d=100.0, max_mc=120.0)


def test_save_range_uses_shared_variables_paths(tmp_path: Path):
    variables = Variables()
    variables.dataset_name = "my_data"
    variables.set_result_data_directory(tmp_path)

    data_tools.save_range(variables)

    assert (tmp_path / "my_data_range.h5").exists()
    assert (tmp_path / "my_data_range.csv").exists()


def test_save_data_can_export_ato(tmp_path: Path):
    variables = Variables()
    variables.dataset_name = "dataset"
    variables.set_result_directory(tmp_path)
    variables.set_result_data_directory(tmp_path)

    data = pd.DataFrame(
        {
            "x (nm)": [1.0, 2.0],
            "y (nm)": [3.0, 4.0],
            "z (nm)": [5.0, 6.0],
            "mc (Da)": [27.0, 28.0],
            "high_voltage (V)": [1000.0, 1100.0],
            "t (ns)": [100.0, 120.0],
            "x_det (cm)": [0.1, 0.2],
            "y_det (cm)": [0.3, 0.4],
            "delta_p": [0, 1],
        }
    )

    data_tools.save_data(data, variables, name="dataset_export", hdf=False, ato_6v=True)

    ato_path = tmp_path / "dataset_export.ato"
    assert ato_path.exists()

    roundtrip = ato_tools.ato_to_ccapt(str(ato_path), mode="ato")
    assert len(roundtrip) == len(data)
    assert list(roundtrip["mc (Da)"]) == pytest.approx([27.0, 28.0])


def test_pos_to_ccapt_uses_standard_pulse_columns(monkeypatch):
    monkeypatch.setattr(
        ccapt_tools.leap_tools,
        "read_pos",
        lambda _path: pd.DataFrame(
            {
                "x (nm)": [1.0, 2.0],
                "y (nm)": [3.0, 4.0],
                "z (nm)": [5.0, 6.0],
                "m/n (Da)": [10.0, 20.0],
            }
        ),
    )

    converted = ccapt_tools.pos_to_ccapt("dummy.pos")

    assert "pulse_v (V)" in converted.columns
    assert "pulse_l (pJ)" in converted.columns
    assert "pulse" not in converted.columns
    assert converted["pulse_v (V)"].tolist() == [0.0, 0.0]
    assert converted["pulse_l (pJ)"].tolist() == [0.0, 0.0]
