import pandas as pd
import pytest

from pyccapt.calibration.data_tools.data_loadcrop import (
    _normalize_extract_mode,
    calculate_ppi_and_ipp,
)


def test_normalize_extract_mode_accepts_legacy_alias():
    assert _normalize_extract_mode("surface_concept") == "dld"
    assert _normalize_extract_mode("roentdec") == "tdc_ro"


def test_normalize_extract_mode_rejects_unknown_value():
    with pytest.raises(ValueError):
        _normalize_extract_mode("invalid_mode")


def test_calculate_ppi_and_ipp_handles_empty_data():
    data = pd.DataFrame({"start_counter": []})
    delta_p, multi = calculate_ppi_and_ipp(data, max_start_counter=1024)
    assert len(delta_p) == 0
    assert len(multi) == 0


def test_calculate_ppi_and_ipp_handles_short_input_without_zero_division():
    data = pd.DataFrame({"start_counter": [1, 1, 2]})
    delta_p, multi = calculate_ppi_and_ipp(data, max_start_counter=1024)

    assert len(delta_p) == 3
    assert len(multi) == 3
