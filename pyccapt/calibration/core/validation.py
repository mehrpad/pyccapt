"""Validation helpers used by calibration workflows."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from pyccapt.calibration.core.exceptions import CalibrationInputError

CALIBRATION_MODES = {"tof", "mc"}
VOLTAGE_MODES = {"ion_seq", "voltage"}
SAMPLE_METHODS = {"histogram", "mean", "median"}
BOWL_SAMPLE_METHODS = {"histogram", "mean"}
SAMPLING_MODES = {"polar", "cartesian"}
SAMPLING_MODE_ALIASES = {
    "polar": "polar",
    "radial": "polar",
    "cartesian": "cartesian",
    "rect": "cartesian",
    "grid": "cartesian",
}
VOLTAGE_MODEL_ALIASES = {
    "poly": "curve_fit",
    "curve_fit": "curve_fit",
    "robust_fit": "robust_fit",
}
BOWL_FIT_MODES = {"curve_fit", "ml_fit", "robust_fit"}


def ensure_choice(value: str, *, field_name: str, allowed: Iterable[str]) -> str:
    """Return `value` if it is in `allowed`, otherwise raise an input error."""
    allowed_values = tuple(allowed)
    if value not in allowed_values:
        raise CalibrationInputError(f"Invalid {field_name!r}: {value!r}. Allowed values are: {', '.join(allowed_values)}")
    return value


def ensure_positive(value: float, *, field_name: str, allow_zero: bool = False) -> float:
    """Validate that `value` is positive (or non-negative when `allow_zero=True`)."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationInputError(f"{field_name!r} must be numeric, got {value!r}") from exc

    if allow_zero and numeric < 0:
        raise CalibrationInputError(f"{field_name!r} must be >= 0, got {numeric}")
    if not allow_zero and numeric <= 0:
        raise CalibrationInputError(f"{field_name!r} must be > 0, got {numeric}")
    return numeric


def ensure_non_empty_array(values: Sequence[float] | np.ndarray, *, field_name: str) -> np.ndarray:
    """Validate that `values` is a non-empty numpy array and return it."""
    arr = np.asarray(values)
    if arr.size == 0:
        raise CalibrationInputError(f"{field_name!r} cannot be empty")
    return arr


def ensure_matching_lengths(*arrays: Sequence[float] | np.ndarray, field_names: Sequence[str]) -> None:
    """Validate that all arrays have identical lengths."""
    lengths = [len(np.asarray(arr)) for arr in arrays]
    if len(set(lengths)) > 1:
        parts = [f"{name}={length}" for name, length in zip(field_names, lengths)]
        raise CalibrationInputError("Mismatched array lengths: " + ", ".join(parts))


def normalize_voltage_model(model: str) -> str:
    """Normalize voltage model names to supported internal identifiers."""
    normalized = VOLTAGE_MODEL_ALIASES.get(model)
    if normalized is None:
        allowed = ", ".join(VOLTAGE_MODEL_ALIASES.keys())
        raise CalibrationInputError(f"Invalid 'model': {model!r}. Allowed values are: {allowed}")
    return normalized


def normalize_sampling_mode(mode: str | None) -> str:
    """Normalize sampling mode names to supported internal identifiers."""
    if mode is None:
        return "polar"
    normalized = SAMPLING_MODE_ALIASES.get(str(mode).strip().lower())
    if normalized is None:
        allowed = ", ".join(sorted(SAMPLING_MODES))
        raise CalibrationInputError(f"Invalid 'sampling_mode': {mode!r}. Allowed values are: {allowed}")
    return normalized
