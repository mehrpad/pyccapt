"""Shared controls and masks for ranged and unranged 3-D ions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from pyccapt.calibration.core.range_assignment import assign_intervals, range_row_definition


UNRANGED = "unranged"


def _row_elements(value):
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [str(item) for item in value]
    return [str(value)]


def is_unranged_row(elements, ion=None):
    """Return whether a range-table row is the synthetic unranged placeholder."""
    labels = _row_elements(elements)
    if ion is not None:
        labels.append(str(ion))
    return any(label.strip().strip("$").lower() in {UNRANGED, 'un', 'unknown'} for label in labels)


def default_element_controls(range_data, value):
    """Build notebook dictionary defaults, always including ``unranged``."""
    controls = {}
    if range_data is not None and not range_data.empty and "element" in range_data:
        for elements in range_data["element"]:
            for element in _row_elements(elements):
                if element.lower() != UNRANGED:
                    controls.setdefault(element, float(value))
    controls[UNRANGED] = float(value)
    return controls


def resolve_element_controls(range_data, controls, default, unranged_default=None):
    """Resolve an element-keyed dictionary to range-row values plus unranged."""
    if not isinstance(controls, Mapping):
        raise ValueError("Element controls must be a dictionary")

    normalized = {
        (UNRANGED if str(key).strip().lower() == UNRANGED else str(key).strip()): _unit_value(value, str(key))
        for key, value in controls.items()
    }
    row_values = []
    for elements in range_data["element"]:
        value = float(default)
        row_elements = _row_elements(elements)
        for element in row_elements:
            if element in normalized:
                value = normalized[element]
        row_values.append(_unit_value(value, row_elements[-1] if row_elements else "range"))
    if unranged_default is None:
        unranged_default = default
    return row_values, normalized.get(UNRANGED, _unit_value(unranged_default, UNRANGED))


def range_row_masks_and_unranged(mc, range_data):
    """Return exclusive row masks using [low, high) and first-row overlap priority."""
    mc = np.asarray(mc)
    lower = np.full(len(range_data), np.nan)
    upper = np.full(len(range_data), np.nan)
    valid_indices = []
    for position, (_, row) in enumerate(range_data.iterrows()):
        definition = range_row_definition(row)
        if definition is None:
            continue
        _, _, lo, hi = definition
        lower[position], upper[position] = lo, hi
        valid_indices.append(position)
    assigned = assign_intervals(mc, lower, upper)
    masks = [assigned == position for position in range(len(range_data))]
    return masks, assigned < 0, valid_indices


def _unit_value(value, label):
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1")
    return value
