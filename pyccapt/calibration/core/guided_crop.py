"""Explicit event selection while retaining the full acquisition for depth prefixes."""
from __future__ import annotations

import numpy as np

from pyccapt.calibration.core.exceptions import CalibrationInputError


def crop_mask(frame, *, event_start=0, event_stop=None, tof_bounds=None,
              voltage_bounds=None, detector_circle=None):
    """Return an aligned mask and JSON-safe criteria; never reorder or delete input.

    Event positions and scalar windows are lower-inclusive/upper-exclusive.
    Detector circles include their boundary. Requested unavailable fields fail
    explicitly; unrelated missing fields do not prevent an event-index crop.
    """
    n = len(frame)
    event_stop = n if event_stop is None else event_stop
    if (not isinstance(event_start, (int, np.integer)) or not isinstance(event_stop, (int, np.integer)) or
            not 0 <= event_start < event_stop <= n):
        raise CalibrationInputError(f"Choose event positions with 0 <= start < stop <= {n}")
    selected = np.zeros(n, dtype=bool)
    selected[event_start:event_stop] = True
    criteria = {"event_start": int(event_start), "event_stop": int(event_stop),
                "interval_convention": "lower inclusive, upper exclusive; detector circle includes boundary"}

    def values(column):
        if column not in frame or column in frame.attrs.get("pyccapt_unavailable_columns", []):
            raise CalibrationInputError(f"Cannot crop on unavailable source field {column}")
        return frame[column].to_numpy(dtype=float)

    for column, name, bounds in (("t (ns)", "tof_bounds", tof_bounds),
                                  ("high_voltage (V)", "voltage_bounds", voltage_bounds)):
        if bounds is None:
            continue
        limits = np.asarray(bounds, dtype=float)
        if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise CalibrationInputError(f"{name} must have finite increasing lower and upper bounds")
        data = values(column)
        selected &= np.isfinite(data) & (data >= limits[0]) & (data < limits[1])
        criteria[name] = limits.tolist()
    if detector_circle is not None:
        circle = np.asarray(detector_circle, dtype=float)
        if circle.shape != (3,) or not np.isfinite(circle).all() or circle[2] <= 0:
            raise CalibrationInputError("Detector circle needs finite x/y center and positive radius in cm")
        x, y = values("x_det (cm)"), values("y_det (cm)")
        selected &= np.isfinite(x) & np.isfinite(y) & (np.hypot(x-circle[0], y-circle[1]) <= circle[2])
        criteria["detector_circle"] = circle.tolist()
    if not selected.any():
        raise CalibrationInputError("The crop keeps no events; widen or reset the selection")
    return selected, criteria
