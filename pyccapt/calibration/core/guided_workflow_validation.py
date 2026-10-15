"""Validate a complete resume payload before touching a live notebook workspace."""
from __future__ import annotations

import numpy as np

from pyccapt.calibration.core.exceptions import CalibrationStateError
from pyccapt.calibration.core.range_assignment import assign_intervals
from pyccapt.calibration.core.ranging_review import PeakReviewQueue


def validate_restored_workflow(workflow):
    """Reject misaligned outputs, broken dependencies and incompatible review state."""
    n = workflow.n

    def vector(value, shape, name, *, positive=False):
        array = np.asarray(value)
        if array.shape != shape or array.dtype.kind not in "iuf" or np.isinf(array).any():
            raise CalibrationStateError(f"Saved {name} is not an aligned finite-or-NaN array")
        if positive and (not np.any(array > 0) or np.any(array[np.isfinite(array)] < 0)):
            raise CalibrationStateError(f"Saved {name} has no valid positive output")
        return array

    def calibration(payload):
        vector(payload["mass"], (n,), "mass", positive=True)
        if payload.get("tof") is not None:
            vector(payload["tof"], (n,), "ToF", positive=True)
        learned = payload.get("model_state")
        if learned and "learned_model" in learned:
            from pyccapt.calibration.data_tools.model_state import restore_model_state
            restore_model_state(learned["learned_model"])
            masks = learned.get("roles", {})
            coverage = np.zeros(n, dtype=np.uint8)
            for mask in masks.values():
                mask = np.asarray(mask)
                if mask.shape != (n,) or mask.dtype != bool:
                    raise CalibrationStateError("Saved model role masks are not aligned booleans")
                coverage += mask
            if np.any(coverage > 1):
                raise CalibrationStateError("Saved model roles overlap")
            if "apply_mask" in learned:
                mask = np.asarray(learned["apply_mask"])
                if mask.shape != (n,) or mask.dtype != bool:
                    raise CalibrationStateError("Saved model apply mask is not aligned")
            if "reference_labels" in learned:
                vector(learned["reference_labels"], (n,), "reference labels")

    def ranges(payload, mass):
        assignment = np.asarray(payload["assignment"])
        rows = payload["rows"]
        if assignment.shape != (n,) or assignment.dtype.kind != "i":
            raise CalibrationStateError("Saved range assignments are not aligned integers")
        actual = assign_intervals(mass, [r["mc_low"] for r in rows], [r["mc_up"] for r in rows])
        if not np.array_equal(actual, assignment):
            raise CalibrationStateError("Saved assignments do not match the accepted range intervals")
        reviewed = PeakReviewQueue.from_state(mass, payload["review_state"])
        if reviewed.ranges().to_dict("records") != rows:
            raise CalibrationStateError("Saved accepted ranges differ from reviewed decisions")

    def accepted(state, review_state=None):
        if set(state) != {"calibration", "ranges", "reconstruction", "analysis"}:
            raise CalibrationStateError("Saved accepted stages are incomplete")
        cal, ranged, spatial = state["calibration"], state["ranges"], state["reconstruction"]
        if (ranged is not None and cal is None) or ((spatial is not None or state["analysis"] is not None) and ranged is None):
            raise CalibrationStateError("Saved workflow dependencies are inconsistent")
        if cal is not None:
            calibration(cal)
        if ranged is not None:
            ranges(ranged, cal["mass"])
        if spatial is not None:
            vector(spatial["xyz"], (n, 3), "reconstruction")
        if review_state is not None:
            if cal is None:
                raise CalibrationStateError("Saved review has no accepted calibration")
            PeakReviewQueue.from_state(cal["mass"], review_state)

    try:
        accepted(workflow.accepted)
        for snapshot in workflow.history:
            accepted(snapshot["accepted"], snapshot["review"])
        for stage, payload in workflow.preview.items():
            if stage == "calibration":
                calibration(payload)
            elif stage == "ranges":
                ranges(payload, workflow.accepted["calibration"]["mass"])
            elif stage == "reconstruction":
                vector(payload["xyz"], (n, 3), "reconstruction preview")
            else:
                raise CalibrationStateError("Unknown saved preview stage")
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationStateError(f"Invalid saved workflow payload: {exc}") from exc
