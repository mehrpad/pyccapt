import pandas as pd
import pytest
import inspect

from pyccapt.calibration.data_tools import data_tools


def test_slice_data_for_export_uses_inclusive_end_index():
    data = pd.DataFrame({"value": [10, 11, 12, 13, 14]})

    sliced = data_tools._slice_data_for_export(data, start_index=1, end_index=3)

    assert sliced["value"].tolist() == [11, 12, 13]


def test_slice_data_for_export_supports_minus_one_for_last_row():
    data = pd.DataFrame({"value": [10, 11, 12]})

    sliced = data_tools._slice_data_for_export(data, start_index=1, end_index=-1)

    assert sliced["value"].tolist() == [11, 12]


def test_slice_data_for_export_rejects_reversed_bounds():
    data = pd.DataFrame({"value": [10, 11, 12]})

    with pytest.raises(ValueError, match="end_index must be greater than or equal to start_index"):
        data_tools._slice_data_for_export(data, start_index=2, end_index=1)


def test_save_data_uses_widget_friendly_export_defaults():
    signature = inspect.signature(data_tools.save_data)

    assert signature.parameters["start_index"].default == 0
    assert signature.parameters["end_index"].default == -1
