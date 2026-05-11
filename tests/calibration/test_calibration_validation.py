import pytest

from pyccapt.calibration.core.exceptions import CalibrationInputError
from pyccapt.calibration.core.validation import (
    ensure_choice,
    ensure_positive,
    normalize_voltage_model,
)


def test_normalize_voltage_model_accepts_poly_alias():
    assert normalize_voltage_model("poly") == "curve_fit"


def test_normalize_voltage_model_rejects_unknown_value():
    with pytest.raises(CalibrationInputError):
        normalize_voltage_model("hybrid")


def test_ensure_choice_rejects_invalid_value():
    with pytest.raises(CalibrationInputError):
        ensure_choice("bad", field_name="mode", allowed=("a", "b"))


def test_ensure_positive_rejects_zero_when_zero_not_allowed():
    with pytest.raises(CalibrationInputError):
        ensure_positive(0, field_name="bin")
