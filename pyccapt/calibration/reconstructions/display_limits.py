"""Display-only masks: retain full reconstruction coordinates and evaporation history."""
from __future__ import annotations

import numpy as np

from pyccapt.calibration.core.exceptions import CalibrationInputError


def display_selection(variables, size):
    value = getattr(variables, 'reconstruction_selection_mask', None)
    if value is None:
        return np.ones(size, dtype=bool)
    value = np.asarray(value)
    if value.shape != (size,) or value.dtype.kind != 'b':
        raise CalibrationInputError('Reconstruction display selection must be boolean and aligned to source events')
    return value


def limit_display(mask, sampled_masks, cluster_result, cluster_only, maximum):
    """Cap the union of species and cluster points, including sparse overlays."""
    mask = np.array(mask, dtype=bool, copy=True)
    plotted = np.logical_or.reduce(sampled_masks) if sampled_masks else mask.copy()
    if cluster_result is not None:
        labels = np.asarray(cluster_result.labels)
        if labels.shape != mask.shape:
            raise CalibrationInputError('Cluster labels must align to source events')
        plotted |= mask if cluster_only else (mask & (labels >= 0))
    if maximum is not None:
        if isinstance(maximum, bool) or not isinstance(maximum, (int, np.integer)) or maximum < 1:
            raise CalibrationInputError('Max displayed points must be a positive integer')
        positions = np.flatnonzero(plotted)
        if len(positions) > maximum:
            chosen = np.zeros(len(mask), dtype=bool)
            chosen[np.random.default_rng(0).choice(positions, maximum, replace=False)] = True
            mask &= chosen
            plotted &= chosen
            sampled_masks = [sample & chosen for sample in sampled_masks]
    return mask, sampled_masks, plotted
