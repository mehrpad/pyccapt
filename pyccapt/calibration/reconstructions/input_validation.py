"""Validation shared by the two reconstruction models."""

import numpy as np

from pyccapt.calibration.core.exceptions import CalibrationInputError


def validate_reconstruction_inputs(detx, dety, hv, flight_path_length, kf, det_eff, icf, field_evap, avg_dens):
    """Reject missing voltage rather than silently changing cumulative depth.

    Missing detector coordinates are allowed for partial hits: these events
    contribute to cumulative volume, but cannot receive finite xyz positions.
    Every event must have positive finite voltage. Callers must repair missing
    voltage from acquisition metadata before reconstructing; rows are not dropped.
    """
    detx, dety, hv = (np.asarray(values, dtype=float) for values in (detx, dety, hv))
    if any(values.ndim != 1 for values in (detx, dety, hv)) or not (detx.size == dety.size == hv.size):
        raise CalibrationInputError('Detector coordinates and voltage must be aligned one-dimensional arrays')
    if not hv.size or np.any(~np.isfinite(hv) | (hv <= 0)):
        raise CalibrationInputError('Every event needs positive finite voltage for cumulative reconstruction; repair missing voltage first')
    if np.isinf(detx).any() or np.isinf(dety).any():
        raise CalibrationInputError('Detector coordinates may contain NaN partial hits, but not infinity')
    radius = np.hypot(detx, dety)
    if not np.isfinite(radius).any() or np.nanmax(radius) <= 0:
        raise CalibrationInputError('Reconstruction requires a nonzero finite detector extent')
    parameters = (flight_path_length, kf, det_eff, icf, field_evap, avg_dens)
    if not np.isfinite(parameters).all() or any(value <= 0 for value in parameters) or det_eff > 1:
        raise CalibrationInputError('Reconstruction parameters must be positive and finite; detection efficiency must be <= 1')
    return detx, dety, hv
