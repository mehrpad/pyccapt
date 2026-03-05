from pathlib import Path

import pandas as pd
import pytest

from pyccapt.calibration.core.share_variables import Variables
from pyccapt.calibration.data_tools import data_tools


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

