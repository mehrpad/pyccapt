"""Custom exceptions used across calibration modules."""


class CalibrationError(Exception):
    """Base class for calibration-related errors."""


class CalibrationInputError(CalibrationError, ValueError):
    """Raised when caller input is invalid for a calibration workflow."""


class CalibrationStateError(CalibrationError, RuntimeError):
    """Raised when shared calibration state is incomplete or inconsistent."""
