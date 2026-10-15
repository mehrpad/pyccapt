"""Canonical mass-window assignment shared by analysis and display.

Windows are half-open ``[mc_low, mc_up)``. For overlapping windows the first
valid row in table order owns the event, matching concentration-profile
semantics. Non-finite masses and invalid windows remain unranged. These rules
prevent one ion appearing in multiple plotted species while being counted once.
"""

from __future__ import annotations

import numpy as np
import ast


def range_row_definition(row, *, require_counts=False):
    """Parse one real species consistently for display and composition analysis.

    Display-only tables may omit atom counts; when present, invalid counts do
    not own events. Synthetic unranged rows never take priority over species.
    """
    def sequence(value):
        if isinstance(value, str) and value.strip().startswith(("[", "(")):
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return [value]
        return list(value) if isinstance(value, (tuple, list, np.ndarray)) else [value]

    elements = tuple(str(value).strip() for value in sequence(row.get("element", [])))
    if any(value.strip("$").lower() in {"un", "unranged", "unknown"}
           for value in (*elements, str(row.get("ion", "")).strip())):
        return None
    if not elements or (require_counts and "complex" not in row):
        return None
    try:
        counts = tuple(float(value) for value in sequence(row.get("complex", [1.] * len(elements))))
        lower, upper = float(row["mc_low"]), float(row["mc_up"])
    except (TypeError, ValueError, KeyError):
        return None
    if (len(elements) != len(counts) or not np.isfinite([lower, upper, *counts]).all()
            or upper <= lower or any(count <= 0 for count in counts) or not counts):
        return None
    return elements, counts, lower, upper


def assign_intervals(mc, lower, upper):
    """Return the original interval index per mass, or -1 when unassigned."""
    mc = np.asarray(mc, dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lower.ndim != 1 or lower.shape != upper.shape:
        raise ValueError('Lower and upper range bounds must be aligned vectors')
    assigned = np.full(mc.shape, -1, dtype=np.int32)
    indices = np.flatnonzero(np.isfinite(lower) & np.isfinite(upper) & (upper > lower))
    if not len(indices):
        return assigned
    order = indices[np.argsort(lower[indices], kind='stable')]
    if np.all(upper[order[:-1]] <= lower[order[1:]]):
        candidates = np.searchsorted(lower[order], mc, side='right') - 1
        safe = np.clip(candidates, 0, len(order) - 1)
        valid = np.isfinite(mc) & (candidates >= 0) & (mc < upper[order[safe]])
        assigned[valid] = order[safe[valid]]
    else:
        available = np.isfinite(mc)
        for index in indices:
            matched = available & (mc >= lower[index]) & (mc < upper[index])
            assigned[matched] = index
            available[matched] = False
    return assigned
