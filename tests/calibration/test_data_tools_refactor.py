from pathlib import Path

import pandas as pd
import pytest

from pyccapt.calibration.core.share_variables import Variables
from pyccapt.calibration.data_tools import ato_tools, data_tools


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
