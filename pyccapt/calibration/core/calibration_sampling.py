"""Shared sampling primitives for legacy and advanced calibration.

Kept independent of workflow modules so either public API can be imported first.
"""

import fast_histogram
import numpy as np
from scipy.signal import find_peaks

from pyccapt.calibration.core.parallel import parallel_map
from pyccapt.calibration.core.validation import (
    SAMPLE_METHODS, ensure_choice, ensure_non_empty_array, normalize_sampling_mode,
)


def _voltage_segments(voltage, values, step):
    """Group voltage intervals once, retaining the original order in each bin.

    Search the same computed boundaries used by the former repeated masks;
    floor((V - Vmin) / step) can disagree at floating-point bin boundaries.
    The final interval includes its upper endpoint.
    """
    v_min, v_max = np.min(voltage), np.max(voltage)
    count = max(1, int(np.ceil(float(v_max - v_min) / float(step))))
    boundaries = v_min + np.arange(1, count) * step
    bins = np.searchsorted(boundaries, voltage, side="right")
    order = np.argsort(bins, kind="stable")
    starts = np.searchsorted(bins[order], np.arange(count + 1))
    sorted_voltage, sorted_values = voltage[order], values[order]
    return [(sorted_voltage[a:b], sorted_values[a:b]) for a, b in zip(starts[:-1], starts[1:])]

def _build_histogram_bins(values, bin_size):
    """Build stable histogram bin edges/count from the actual data span."""
    data = ensure_non_empty_array(values, field_name="histogram_values")
    lower = float(np.min(data))
    upper = float(np.max(data))
    span = max(upper - lower, float(bin_size))
    n_bins = max(10, int(np.ceil(span / float(bin_size))))
    edges = np.linspace(lower, upper, n_bins + 1)
    return edges, n_bins

def _resolve_peak_location(values, method, bin_size, fast_calibration=False, rng=None):
    """Resolve the reference peak location by histogram/mean/median.

    Parameters
    ----------
    rng : optional ``numpy.random.Generator``
        Generator used by the ``fast_calibration`` subsample path. Defaults
        to a seeded generator so the same input produces the same peak
        location across runs (the previous implementation used the global
        ``np.random`` state, making fast-mode calibrations non-reproducible).
    """
    ensure_choice(method, field_name="maximum_cal_method", allowed=SAMPLE_METHODS)
    data = ensure_non_empty_array(values, field_name="peak_values")
    if fast_calibration and data.size > 10:
        if rng is None:
            rng = np.random.default_rng(0)
        data = rng.choice(data, int(data.size * 0.1), replace=False)

    if method == "mean":
        return float(np.mean(data))
    if method == "median":
        return float(np.median(data))

    # A constant population has an exact peak location but no nonzero-width
    # histogram domain. This also occurs in small/quantized voltage bins.
    if np.min(data) == np.max(data):
        return float(data[0])

    bins, n_bins = _build_histogram_bins(data, bin_size)
    hist = fast_histogram.histogram1d(
        data,
        bins=n_bins,
        range=(np.min(data), np.max(data)),
    )
    peaks, properties = find_peaks(hist, height=0)
    if len(peaks) == 0:
        return float(np.mean(data))
    index_peak_max_ini = np.argmax(properties["peak_heights"])
    # ``bins`` is the edges array (length n_bins+1); ``peaks`` indexes the
    # counts array (length n_bins). bins[peak] returns the LEFT edge of the
    # bin, biasing the peak location low by half a bin width. Use the bin
    # center instead.
    peak_idx = int(peaks[index_peak_max_ini])
    return float(0.5 * (bins[peak_idx] + bins[peak_idx + 1]))

def _radial_bowl_corr(data_xy, a, b, c, d, e):
    """Radial-dominant bowl model where r^2 drives the primary curvature."""
    x = np.asarray(data_xy[0], dtype=float)
    y = np.asarray(data_xy[1], dtype=float)
    r2 = x**2 + y**2
    return a + b * r2 + c * (r2**2) + d * x + e * y

def _resolve_sampling_mode(sampling_mode, variables=None):
    """Resolve sampling mode with a variable-level default fallback."""
    configured = sampling_mode
    if configured is None and variables is not None:
        configured = getattr(variables, "bowl_sampling_mode", "polar")
    mode = normalize_sampling_mode(configured)
    if variables is not None and hasattr(variables, "bowl_sampling_mode"):
        variables.bowl_sampling_mode = mode
    return mode

def _cell_peak_value(values, maximum_location, sample_range_max, bin_size):
    if sample_range_max == 'mean':
        return float(np.mean(values)) / maximum_location
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float('nan')
    value_span = float(np.max(values) - np.min(values))
    if not np.isfinite(value_span) or value_span <= 0:
        return float(np.mean(values)) / maximum_location
    bins, n_bins = _build_histogram_bins(values, bin_size)
    y_hist = fast_histogram.histogram1d(
        values,
        bins=n_bins,
        range=(np.min(values), np.max(values)),
    )
    peaks, properties = find_peaks(y_hist, height=0)
    if len(peaks) == 0:
        return float(np.mean(values)) / maximum_location
    index_peak_max_ini = np.argmax(properties['peak_heights'])
    # ``bins`` is the edges array (length n_bins+1); ``peaks`` indexes the
    # counts array. Use the bin CENTER, not the left edge.
    peak_idx = int(peaks[index_peak_max_ini])
    peak_center = 0.5 * (bins[peak_idx] + bins[peak_idx + 1])
    return float(peak_center) / maximum_location

def _iter_polar_cells(radial_distance, sample_size, det_diam):
    r_max_data = float(np.max(radial_distance)) if len(radial_distance) else 0.0
    r_max_detector = max(0.0, float(det_diam) / 2.0)
    r_max = max(r_max_data, r_max_detector, float(sample_size))
    n_rings = max(1, int(np.ceil(r_max / float(sample_size))))
    ring_edges = np.linspace(0.0, r_max, n_rings + 1)

    for r_idx in range(n_rings):
        r_min_i = float(ring_edges[r_idx])
        r_max_i = float(ring_edges[r_idx + 1])
        r_mid = 0.5 * (r_min_i + r_max_i)
        circumference = 2.0 * np.pi * max(r_mid, 0.5 * float(sample_size))
        n_sectors = int(np.clip(np.ceil(circumference / float(sample_size)), 8, 96))
        for theta_idx in range(n_sectors):
            theta_min = float(theta_idx * 2.0 * np.pi / n_sectors)
            theta_max = float((theta_idx + 1) * 2.0 * np.pi / n_sectors)
            yield r_min_i, r_max_i, theta_min, theta_max

def _collect_spatial_samples(
    dld_x,
    dld_y,
    dld_t,
    maximum_location,
    sample_range_max,
    sample_size,
    bin_size,
    *,
    sampling_mode,
    det_diam,
    dld_v=None,
):
    # The inner work for each cell is a histogram + peak find + mean / median
    # call on a small subarray. Every cell is independent of every other, and
    # the inner kernel is NumPy/SciPy (GIL-releasing), so this is the textbook
    # case for parallel_map(gil_releasing=True). On a 80 mm detector at 2 mm
    # sampling we get O(2000) cells per call -- typically 4-8x speedup with
    # threads, automatic serial fallback for tiny inputs.
    if sampling_mode == 'polar':
        radial_distance = np.hypot(dld_x, dld_y)
        polar_angle = np.mod(np.arctan2(dld_y, dld_x), 2.0 * np.pi)
        cells = list(_iter_polar_cells(radial_distance, sample_size, det_diam))

        def _eval_polar_cell(bounds):
            r_min_i, r_max_i, theta_min, theta_max = bounds
            mask = (
                (radial_distance >= r_min_i)
                & (radial_distance < r_max_i)
                & (polar_angle >= theta_min)
                & (polar_angle < theta_max)
            )
            if not np.any(mask):
                return None
            cell_t = dld_t[mask]
            normalized_value = _cell_peak_value(cell_t, maximum_location, sample_range_max, bin_size)
            if not np.isfinite(normalized_value) or normalized_value <= 0:
                return None
            row = (
                float(np.median(dld_x[mask])),
                float(np.median(dld_y[mask])),
                normalized_value,
                float(np.mean(dld_v[mask])) if dld_v is not None else None,
            )
            return row

        results = parallel_map(_eval_polar_cell, cells, gil_releasing=True)
    else:
        cell_size = float(sample_size)
        x_min = float(np.floor(np.min(dld_x)))
        x_max = float(np.ceil(np.max(dld_x)))
        y_min = float(np.floor(np.min(dld_y)))
        y_max = float(np.ceil(np.max(dld_y)))
        n_cols = max(1, int(np.ceil((x_max - x_min) / cell_size)))
        n_rows = max(1, int(np.ceil((y_max - y_min) / cell_size)))
        x_bin = np.floor((dld_x - x_min) / cell_size).astype(int)
        y_bin = np.floor((dld_y - y_min) / cell_size).astype(int)
        valid = (x_bin >= 0) & (x_bin < n_cols) & (y_bin >= 0) & (y_bin < n_rows)
        if not np.any(valid):
            results: list = []
        else:
            cell_id = y_bin[valid] * n_cols + x_bin[valid]
            order = np.argsort(cell_id, kind='mergesort')
            x_sorted = dld_x[valid][order]
            y_sorted = dld_y[valid][order]
            t_sorted = dld_t[valid][order]
            v_sorted = dld_v[valid][order] if dld_v is not None else None
            cell_sorted = cell_id[order]
            _, starts, counts = np.unique(cell_sorted, return_index=True, return_counts=True)
            # Stays serial: cartesian cells operate on pre-sorted contiguous
            # slices so each cell is sub-millisecond. Thread dispatch overhead
            # exceeds the per-cell work, so parallelism here regresses
            # wall-clock time on every realistic dataset.
            results = []
            for start, count in zip(starts, counts):
                stop = start + count
                cell_t = t_sorted[start:stop]
                normalized_value = _cell_peak_value(cell_t, maximum_location, sample_range_max, bin_size)
                if not np.isfinite(normalized_value) or normalized_value <= 0:
                    results.append(None)
                    continue
                results.append(
                    (
                        float(np.median(x_sorted[start:stop])),
                        float(np.median(y_sorted[start:stop])),
                        normalized_value,
                        float(np.mean(v_sorted[start:stop])) if v_sorted is not None else None,
                    )
                )

    x_samples: list = []
    y_samples: list = []
    t_samples: list = []
    v_samples: list = []
    for row in results:
        if row is None:
            continue
        x_samples.append(row[0])
        y_samples.append(row[1])
        t_samples.append(row[2])
        if dld_v is not None:
            v_samples.append(row[3])

    result = {
        'x': np.asarray(x_samples, dtype=float),
        'y': np.asarray(y_samples, dtype=float),
        't': np.asarray(t_samples, dtype=float),
    }
    if dld_v is not None:
        result['v'] = np.asarray(v_samples, dtype=float)
    return result
