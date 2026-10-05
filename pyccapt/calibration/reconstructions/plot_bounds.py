"""Robust plot-bound helpers for reconstruction visualizations."""

from __future__ import annotations

import numpy as np


def evaporation_frame_boundaries(num_events, frame_count):
    """Return evenly spaced cumulative event endpoints for an evaporation GIF."""
    num_events = int(num_events)
    frame_count = int(frame_count)
    if frame_count < 1:
        raise ValueError("Evaporation GIF frame count must be at least 1")
    if num_events < 1:
        return np.array([], dtype=int)
    count = min(num_events, frame_count)
    return np.unique(np.ceil(np.linspace(0, num_events, count + 1)[1:]).astype(int))


def sample_mask(mask, fraction, n_points):
    """Return a deterministic random subset mask for a display fraction."""
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != int(n_points):
        raise ValueError("mask length must match n_points")
    true_indices = np.flatnonzero(mask)
    size = int(len(true_indices) * float(fraction))
    size = max(0, min(size, len(true_indices)))
    sampled_mask = np.zeros(int(n_points), dtype=bool)
    if size > 0:
        sampled_indices = np.random.default_rng(int(len(true_indices))).choice(
            true_indices,
            size=size,
            replace=False,
        )
        sampled_mask[sampled_indices] = True
    return sampled_mask


def axis_range_from_mask(values, mask, fallback_mask=None, lower_percentile=0.1, upper_percentile=99.9):
    """Return robust finite bounds for one coordinate axis."""
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    finite_mask = mask & np.isfinite(values)
    if not np.any(finite_mask) and fallback_mask is not None:
        finite_mask = np.asarray(fallback_mask, dtype=bool) & np.isfinite(values)
    if not np.any(finite_mask):
        finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return [0.0, 1.0]

    selected = values[finite_mask]
    if selected.size >= 3:
        low, high = np.nanpercentile(selected, [lower_percentile, upper_percentile])
    else:
        low, high = np.nanmin(selected), np.nanmax(selected)

    if not np.isfinite(low) or not np.isfinite(high):
        low, high = np.nanmin(selected), np.nanmax(selected)
    if low == high:
        padding = max(abs(float(low)) * 0.01, 1.0)
    else:
        padding = 0.02 * (float(high) - float(low))
    return [float(low) - padding, float(high) + padding]


def range_cube_from_mask(variables, mask, fallback_mask=None):
    """Build an x/y/z cube from finite plotted points, clipped against outliers."""
    return [
        axis_range_from_mask(variables.x, mask, fallback_mask=fallback_mask),
        axis_range_from_mask(variables.y, mask, fallback_mask=fallback_mask),
        axis_range_from_mask(variables.z, mask, fallback_mask=fallback_mask),
    ]
