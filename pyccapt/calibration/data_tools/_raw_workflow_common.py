"""Shared constants and detector-agnostic helpers for the raw-data workflow.

This is an internal sibling of :mod:`raw_data_workflow`. The public surface
lives in that module and re-exports everything from here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from pyccapt.calibration.data_tools import data_tools, lazy_io


# ---------------------------------------------------------------------------
# DataFrame / LazyTable union helpers
# ---------------------------------------------------------------------------


# Type alias kept loose so callers don't need to import LazyTable to type-hint
# the public functions; runtime dispatch is via isinstance checks below.
TabularSource = object  # pd.DataFrame | LazyTable


def _is_lazy(source) -> bool:
    return isinstance(source, lazy_io.LazyTable)


def _columns(source: TabularSource):
    return source.columns if _is_lazy(source) else list(source.columns)


def _has_column(source: TabularSource, name: str) -> bool:
    return name in _columns(source)


def _row_count(source: TabularSource) -> int:
    return source.n_rows if _is_lazy(source) else int(len(source))


def _iter_column_chunks(source: TabularSource, column: str, *, chunk_size: int = lazy_io.DEFAULT_CHUNK_SIZE):
    """Yield successive ``ndarray`` chunks of one column, dtype-cast to float64.

    For a DataFrame the full column is yielded as a single chunk (no copy when
    the underlying storage is already float). For a :class:`LazyTable` the
    column is streamed via :meth:`LazyTable.iter_chunks` so peak RAM is
    bounded by ``chunk_size`` rows.
    """
    if _is_lazy(source):
        for chunk in source.iter_chunks([column], chunk_size=chunk_size):
            arr = chunk[column]
            if arr.dtype != np.float64:
                arr = arr.astype(np.float64, copy=False)
            yield arr
    else:
        yield source[column].to_numpy(dtype=float)


def _iter_columns_chunks(
    source: TabularSource,
    columns: Sequence[str],
    *,
    chunk_size: int = lazy_io.DEFAULT_CHUNK_SIZE,
):
    """Yield ``dict[name, ndarray]`` chunks for a list of columns."""
    if _is_lazy(source):
        for chunk in source.iter_chunks(list(columns), chunk_size=chunk_size):
            yield {name: chunk[name] for name in columns}
    else:
        yield {name: source[name].to_numpy(dtype=float) for name in columns}


TOF_FACTOR_NS = 27.432 / (1000.0 * 4.0)
TOF_FACTOR_NS_1D = 27.432 / (1000.0 * 2.0)
DETBINS = 4900
BINNING_FACTOR = 2
XY_FACTOR = 80.0 / DETBINS * BINNING_FACTOR
XY_BIN_SHIFT = DETBINS / BINNING_FACTOR / 2.0

DEFAULT_ROENTDEK_DETX_COLUMNS = (6, 10, 14, 18)
DEFAULT_ROENTDEK_DETY_COLUMNS = (7, 11, 15, 19)
DEFAULT_ROENTDEK_SIGNAL_COLUMNS = (8, 12, 16, 20)

DLTS_COLORS = {
    2: '#d97706',
    4: '#2563eb',
    6: '#059669',
}


# ---------------------------------------------------------------------------
# Detector geometry / TDC bin-to-time-position conversion constants.
# ---------------------------------------------------------------------------
# These default values mirror the historical hard-coded numbers (Surface
# Concept 80 mm detector, 4900 TDC bins, binning factor 2). They can also be
# loaded per-rig from ``pyccapt/config.toml`` via :func:`load_detector_constants`
# so that the same code can be applied to instruments with different TDC
# resolution or detector geometry without touching source.

# Match the SC defaults so existing tests / direct imports keep working.
_SURFACE_CONCEPT_DEFAULT_CONSTANTS = {
    'tof_ns_per_bin': TOF_FACTOR_NS,
    'tof_ns_per_bin_1d': TOF_FACTOR_NS_1D,
    'detector_bins': DETBINS,
    'binning_factor': BINNING_FACTOR,
    'detector_width_mm': 80.0,
    'detector_limit_cm': 4.0,
    'xy_factor': XY_FACTOR,
    'xy_bin_shift': XY_BIN_SHIFT,
}

# RoentDek default — keep the 80 mm geometry so existing fixtures still work.
# The TOF bin width (25 ps) is a typical CTNM4 / CRTM-class TDC value; tune
# via config for your specific board.
_ROENTDEK_DEFAULT_CONSTANTS = {
    'tof_ns_per_bin': 0.025,
    'detector_bins': 4900,
    'binning_factor': 2,
    'detector_width_mm': 80.0,
    'detector_limit_cm': 4.0,
    'xy_factor': 80.0 / 4900.0 * 2.0,
    'xy_bin_shift': 4900.0 / 2.0 / 2.0,
}


def _config_get(conf, key, default):
    """Pull a numeric setting out of a config mapping, ignoring missing keys."""
    if conf is None:
        return default
    try:
        value = conf[key] if hasattr(conf, '__getitem__') else getattr(conf, key, default)
    except (KeyError, AttributeError, TypeError):
        return default
    return value if value is not None else default


def load_detector_constants(detector_kind: str, conf=None) -> dict[str, float]:
    """Return the geometry / time-base constants for one detector family.

    ``detector_kind`` is ``'surface_concept'``, ``'roentdek'``, or
    ``'single_delay_line'``. ``conf`` is an optional mapping (e.g. the dict
    returned by :func:`pyccapt.control.core.read_files.load_config_file`); when
    supplied, per-rig overrides take precedence over the historical defaults.

    The returned dict always carries the derived ``xy_factor`` (mm / bin) and
    ``xy_bin_shift`` (bins) so callers can convert raw delay-line timestamps
    to detector-cm coordinates without re-deriving them from primitive
    constants in three different places.
    """
    kind = (detector_kind or '').lower()
    if kind == 'roentdek':
        defaults = dict(_ROENTDEK_DEFAULT_CONSTANTS)
        prefix = 'ro_'
    elif kind == 'surface_concept' or kind == 'single_delay_line':
        defaults = dict(_SURFACE_CONCEPT_DEFAULT_CONSTANTS)
        prefix = 'sc_'
    else:
        defaults = dict(_SURFACE_CONCEPT_DEFAULT_CONSTANTS)
        prefix = 'sc_'

    constants = {
        'tof_ns_per_bin': float(_config_get(conf, f'{prefix}tof_ns_per_bin', defaults['tof_ns_per_bin'])),
        'detector_bins': int(_config_get(conf, f'{prefix}detector_bins', defaults['detector_bins'])),
        'binning_factor': int(_config_get(conf, f'{prefix}detector_binning_factor', defaults['binning_factor'])),
        'detector_width_mm': float(_config_get(conf, f'{prefix}detector_width_mm', defaults['detector_width_mm'])),
        'detector_limit_cm': float(_config_get(conf, f'{prefix}detector_limit_cm', defaults['detector_limit_cm'])),
    }
    # Re-derive the spatial factors from the primitives so the returned dict
    # is always self-consistent even when the user only overrode part of the
    # geometry.
    constants['xy_factor'] = constants['detector_width_mm'] / constants['detector_bins'] * constants['binning_factor']
    constants['xy_bin_shift'] = constants['detector_bins'] / constants['binning_factor'] / 2.0
    if kind == 'surface_concept' or kind == 'single_delay_line':
        constants['tof_ns_per_bin_1d'] = constants['tof_ns_per_bin'] * 2.0
    return constants


def _normalize_signal_kind(signal_kind: str) -> str:
    value = str(signal_kind).strip().lower()
    if value not in {'tof', 'mc'}:
        raise ValueError(f"signal_kind must be 'tof' or 'mc', got {signal_kind!r}")
    return value


def _signal_column_and_label(signal_kind: str) -> tuple[str, str]:
    value = _normalize_signal_kind(signal_kind)
    if value == 'mc':
        return 'mc (Da)', 'Mass-to-charge [Da]'
    return 'tof (ns)', 'Time of flight [ns]'


def _normalize_index_positions(values: Sequence[int] | str, label: str) -> tuple[int, ...]:
    if isinstance(values, str):
        cleaned = [item.strip() for item in values.split(',') if item.strip()]
        if not cleaned:
            raise ValueError(f'{label} cannot be empty')
        try:
            parsed = tuple(int(item) for item in cleaned)
        except ValueError as exc:
            raise ValueError(f'{label} must contain comma-separated integer column indices') from exc
        return parsed

    try:
        parsed = tuple(int(item) for item in values)
    except TypeError as exc:
        raise ValueError(f'{label} must be a sequence of integers') from exc
    if not parsed:
        raise ValueError(f'{label} cannot be empty')
    return parsed


def normalize_signal_windows(windows: Sequence[dict] | Sequence[Sequence] | None) -> list[dict[str, float | str]]:
    """Validate and normalize optional user-defined peak windows."""
    normalized: list[dict[str, float | str]] = []
    if not windows:
        return normalized

    for index, window in enumerate(windows, start=1):
        if isinstance(window, dict):
            label = window.get('label') or f'Window {index}'
            minimum = window.get('min')
            maximum = window.get('max')
        else:
            if len(window) != 3:
                raise ValueError('Each signal window must provide (label, min, max)')
            label, minimum, maximum = window
            label = label or f'Window {index}'

        try:
            minimum_value = float(minimum)
            maximum_value = float(maximum)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Signal window {label!r} must use numeric limits') from exc
        if maximum_value <= minimum_value:
            raise ValueError(
                f"Signal window {label!r} is invalid: max ({maximum_value}) must be greater than min ({minimum_value})"
            )
        normalized.append({'label': str(label), 'min': minimum_value, 'max': maximum_value})
    normalized.sort(key=lambda item: float(item['min']))
    for previous, current in zip(normalized, normalized[1:]):
        if float(current['min']) < float(previous['max']):
            raise ValueError(
                f"Signal windows {previous['label']!r} and {current['label']!r} overlap. "
                'Use non-overlapping ranges to avoid double-counting.'
            )
    return normalized


def load_numeric_text_table(file_path: str, *, delimiter: str | None = None, skiprows: int = 0) -> np.ndarray:
    """Load a numeric text table used by the raw-data notebooks."""
    table = np.loadtxt(file_path, delimiter=delimiter, skiprows=skiprows)
    return np.atleast_2d(table)


def _validate_numeric_table_columns(table: np.ndarray, columns: Sequence[int], label: str) -> None:
    max_index = max(columns)
    if table.shape[1] <= max_index:
        raise ValueError(f'{label} requires column index {max_index}, but the numeric table only has {table.shape[1]} columns')


def _compute_histogram_bins(values: np.ndarray, bin_size: float) -> np.ndarray:
    if values.size == 0:
        raise ValueError('No values are available for histogram plotting')
    if bin_size <= 0:
        raise ValueError('bin_size must be positive')

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError('Histogram values must be finite')
    if math.isclose(minimum, maximum):
        maximum = minimum + bin_size
    return np.arange(minimum, maximum + bin_size, bin_size)


def summarize_signal_windows(
    hit_table: pd.DataFrame,
    windows: Sequence[dict] | Sequence[Sequence] | None,
    *,
    signal_kind: str = 'tof',
    only_in_detector: bool = True,
    noise_label: str = 'Noise',
) -> pd.DataFrame:
    """Count hits falling into user-defined signal windows for each DLTS class."""
    normalized_windows = normalize_signal_windows(windows)
    if not normalized_windows:
        return pd.DataFrame(columns=['label', 'dlts', 'count'])

    signal_column, _ = _signal_column_and_label(signal_kind)
    if signal_column not in hit_table.columns:
        raise ValueError(f'Column {signal_column!r} is not present in the hit table')

    frame = hit_table.copy()
    if only_in_detector and 'in_detector' in frame.columns:
        frame = frame[frame['in_detector']]
    frame = frame[np.isfinite(frame[signal_column])]
    if frame.empty:
        return pd.DataFrame(columns=['label', 'dlts', 'count'])

    summaries = []
    for dlts in sorted(frame['dlts'].dropna().astype(int).unique()):
        subset = frame[frame['dlts'] == dlts]
        signal_values = subset[signal_column].to_numpy()
        assigned = np.zeros(len(subset), dtype=bool)
        for window in normalized_windows:
            mask = np.logical_and(signal_values >= window['min'], signal_values <= window['max'])
            assigned |= mask
            summaries.append({'label': window['label'], 'dlts': dlts, 'count': int(mask.sum())})
        summaries.append({'label': noise_label, 'dlts': dlts, 'count': int((~assigned).sum())})
    return pd.DataFrame(summaries)


def plot_signal_window_breakdown(
    hit_table: pd.DataFrame,
    windows: Sequence[dict] | Sequence[Sequence] | None,
    *,
    signal_kind: str = 'tof',
    only_in_detector: bool = True,
    title: str | None = None,
) -> plt.Figure | None:
    """Plot grouped counts for user-defined signal windows."""
    summary = summarize_signal_windows(
        hit_table,
        windows,
        signal_kind=signal_kind,
        only_in_detector=only_in_detector,
    )
    if summary.empty:
        return None

    labels = list(dict.fromkeys(summary['label'].tolist()))
    dlts_values = sorted(summary['dlts'].dropna().astype(int).unique())
    fig, ax = plt.subplots(figsize=(max(6.5, len(labels) * 1.4), 3.2))

    width = 0.75 / max(len(dlts_values), 1)
    positions = np.arange(len(labels))
    for offset, dlts in enumerate(dlts_values):
        subset = summary[summary['dlts'] == dlts].set_index('label').reindex(labels, fill_value=0)
        ax.bar(
            positions + (offset - (len(dlts_values) - 1) / 2.0) * width,
            subset['count'].to_numpy(),
            width=width,
            label=f'{dlts} DLTS',
            color=DLTS_COLORS.get(int(dlts), '#6b7280'),
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('Count')
    if summary['count'].max() > 20:
        ax.set_yscale('log')
    if title:
        ax.set_title(title)
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig


def plot_signal_overlay_by_dlts(
    hit_table: pd.DataFrame,
    *,
    signal_kind: str = 'tof',
    max_value: float | None = None,
    bin_size: float = 0.1,
    only_in_detector: bool = True,
    title: str | None = None,
) -> plt.Figure:
    """Overlay histograms for the available DLTS classes."""
    signal_column, axis_label = _signal_column_and_label(signal_kind)
    if signal_column not in hit_table.columns:
        raise ValueError(f'Column {signal_column!r} is not present in the hit table')

    frame = hit_table.copy()
    if only_in_detector and 'in_detector' in frame.columns:
        frame = frame[frame['in_detector']]
    frame = frame[np.isfinite(frame[signal_column])]
    if max_value is not None:
        frame = frame[frame[signal_column] <= max_value]
    if frame.empty:
        raise ValueError('No hits are available for the requested histogram plot')

    bins = _compute_histogram_bins(frame[signal_column].to_numpy(), bin_size)
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    plotted = False
    for dlts in sorted(frame['dlts'].dropna().astype(int).unique()):
        subset = frame[frame['dlts'] == dlts][signal_column].to_numpy()
        if subset.size == 0:
            continue
        ax.hist(
            subset,
            bins=bins,
            histtype='stepfilled',
            alpha=0.45,
            log=True,
            label=f'{dlts} DLTS',
            color=DLTS_COLORS.get(int(dlts), '#6b7280'),
        )
        plotted = True

    if not plotted:
        raise ValueError('No DLTS classes were available for overlay plotting')

    ax.set_xlabel(axis_label)
    ax.set_ylabel('Event count')
    if title:
        ax.set_title(title)
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig


def plot_detector_overview(
    hit_table: pd.DataFrame,
    *,
    detector_limit_cm: float | None = None,
    only_in_detector: bool = False,
    title_prefix: str = 'Detector',
) -> plt.Figure:
    """Plot an overall detector map plus one panel per DLTS class."""
    if 'x_det (cm)' not in hit_table.columns or 'y_det (cm)' not in hit_table.columns:
        raise ValueError("Detector plots require 'x_det (cm)' and 'y_det (cm)' columns")

    frame = hit_table.copy()
    if only_in_detector and 'in_detector' in frame.columns:
        frame = frame[frame['in_detector']]
    frame = frame[np.isfinite(frame['x_det (cm)']) & np.isfinite(frame['y_det (cm)'])]
    if frame.empty:
        raise ValueError('No detector coordinates are available for plotting')

    unique_dlts = sorted(frame['dlts'].dropna().astype(int).unique()) if 'dlts' in frame.columns else []
    panel_keys: list[str | int] = ['All', *unique_dlts]
    fig, axes = plt.subplots(1, len(panel_keys), figsize=(4.1 * len(panel_keys), 3.2), squeeze=False)

    if detector_limit_cm is None:
        detector_extent = float(
            max(
                np.max(np.abs(frame['x_det (cm)'].to_numpy())),
                np.max(np.abs(frame['y_det (cm)'].to_numpy())),
                1.0,
            )
        )
    else:
        detector_extent = float(detector_limit_cm)

    for axis, panel_key in zip(axes[0], panel_keys):
        if panel_key == 'All':
            subset = frame
            panel_title = f'{title_prefix} map'
            color = '#2563eb'
        else:
            subset = frame[frame['dlts'] == panel_key]
            panel_title = f'{panel_key} DLTS'
            color = DLTS_COLORS.get(int(panel_key), '#6b7280')

        if subset.empty:
            axis.text(0.5, 0.5, 'No hits', ha='center', va='center', transform=axis.transAxes)
        elif len(subset) > 1200:
            axis.hexbin(
                subset['x_det (cm)'],
                subset['y_det (cm)'],
                gridsize=50,
                cmap='viridis',
                mincnt=1,
            )
        else:
            axis.scatter(
                subset['x_det (cm)'],
                subset['y_det (cm)'],
                s=4,
                alpha=0.45,
                color=color,
                edgecolors='none',
            )
        axis.set_xlim(-detector_extent, detector_extent)
        axis.set_ylim(-detector_extent, detector_extent)
        axis.set_aspect('equal', adjustable='box')
        axis.set_xlabel('x_det (cm)')
        axis.set_ylabel('y_det (cm)')
        axis.set_title(panel_title)

    fig.tight_layout()
    return fig


def _binned_status_fraction(frame: pd.DataFrame, x_column: str, statuses: list[str], *, num_bins: int = 20) -> pd.DataFrame:
    subset = frame.copy()
    subset = subset[np.isfinite(subset[x_column].to_numpy(dtype=float))]
    if subset.empty:
        return pd.DataFrame(columns=['bin_center', 'status', 'fraction'])

    values = subset[x_column].to_numpy(dtype=float)
    if np.min(values) == np.max(values):
        return pd.DataFrame(columns=['bin_center', 'status', 'fraction'])

    edges = np.linspace(np.min(values), np.max(values), num_bins + 1)
    subset['__bin'] = pd.cut(values, bins=edges, include_lowest=True, duplicates='drop')
    rows = []
    for interval, interval_frame in subset.groupby('__bin', observed=True):
        if interval is None or interval_frame.empty:
            continue
        bin_center = float(interval.mid)
        total = float(len(interval_frame))
        for status in statuses:
            fraction = float(np.count_nonzero(interval_frame['status'] == status)) / total
            rows.append({'bin_center': bin_center, 'status': status, 'fraction': fraction})
    return pd.DataFrame(rows)


def _auto_peak_windows(
    values: np.ndarray, *, max_value: float | None = None, max_peaks: int = 3
) -> list[dict[str, float | str]]:
    finite = values[np.isfinite(values)]
    if max_value is not None:
        finite = finite[finite <= max_value]
    if finite.size < 50:
        return []

    bin_size = max((np.percentile(finite, 95) - np.percentile(finite, 5)) / 600.0, 0.1)
    bins = _compute_histogram_bins(finite, bin_size)
    hist, edges = np.histogram(finite, bins=bins)
    prominence = max(np.max(hist) * 0.05, 5.0)
    peaks, _ = find_peaks(hist, prominence=prominence, distance=max(3, len(hist) // 60))
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(hist))], dtype=int)

    ranked = sorted(peaks, key=lambda index: hist[index], reverse=True)[:max_peaks]
    windows = []
    for peak_index, histogram_index in enumerate(sorted(ranked), start=1):
        center = float(0.5 * (edges[histogram_index] + edges[histogram_index + 1]))
        width = max(2.0 * bin_size * 8.0, center * 0.01)
        windows.append({'label': f'Peak {peak_index}', 'min': center - width / 2.0, 'max': center + width / 2.0})
    return normalize_signal_windows(windows)


def compute_tof_segment_drift(
    source: TabularSource,
    *,
    windows: Sequence[dict] | Sequence[Sequence] | None = None,
    num_segments: int = 20,
    max_value: float | None = None,
    chunk_size: int = lazy_io.DEFAULT_CHUNK_SIZE,
) -> pd.DataFrame:
    """Estimate TOF peak drift by splitting the dataset into time/index segments.

    Accepts either a ``pandas.DataFrame`` or a
    :class:`pyccapt.calibration.data_tools.lazy_io.LazyTable`. With a LazyTable
    only one segment's TOF window is resident at a time (peak RAM ~ segment
    length × 8 B), making drift estimation viable on big raw files.
    """
    tof_column = 'tof (ns)' if _has_column(source, 'tof (ns)') else 't (ns)' if _has_column(source, 't (ns)') else None
    if tof_column is None:
        raise ValueError("A TOF column ('tof (ns)' or 't (ns)') is required for drift analysis")

    # We don't auto-derive windows in lazy mode because that would require a
    # full pre-pass over the column; require the caller to pass them.
    if not windows and _is_lazy(source):
        raise ValueError(
            "compute_tof_segment_drift requires explicit `windows` when called "
            "on a LazyTable (auto-detection would need a separate full pass)."
        )

    n_rows = _row_count(source)
    if n_rows == 0:
        return pd.DataFrame(columns=['segment', 'peak_label', 'peak_position', 'peak_count'])

    if windows:
        normalized_windows = normalize_signal_windows(windows)
    else:
        # Eager path with auto-peak detection.
        values = source[tof_column].to_numpy(dtype=float)
        normalized_windows = _auto_peak_windows(values, max_value=max_value)
    if not normalized_windows:
        return pd.DataFrame(columns=['segment', 'peak_label', 'peak_position', 'peak_count'])

    segment_edges = np.linspace(0, n_rows, num_segments + 1, dtype=int)

    if _is_lazy(source):
        col = source[tof_column]

        def fetch_segment(start: int, stop: int) -> np.ndarray:
            return np.asarray(col[start:stop], dtype=float)
    else:
        full = source[tof_column].to_numpy(dtype=float)

        def fetch_segment(start: int, stop: int) -> np.ndarray:
            return full[start:stop]

    rows = []
    for segment_index in range(num_segments):
        start = int(segment_edges[segment_index])
        stop = int(segment_edges[segment_index + 1])
        if stop <= start:
            continue
        segment_values = fetch_segment(start, stop)
        segment_values = segment_values[np.isfinite(segment_values)]
        if max_value is not None:
            segment_values = segment_values[segment_values <= max_value]
        if segment_values.size == 0:
            continue
        for window in normalized_windows:
            window_values = segment_values[(segment_values >= float(window['min'])) & (segment_values <= float(window['max']))]
            if window_values.size < 5:
                continue
            hist, edges = np.histogram(window_values, bins=max(20, min(80, window_values.size // 5)))
            peak_bin = int(np.argmax(hist))
            peak_position = float(0.5 * (edges[peak_bin] + edges[peak_bin + 1]))
            rows.append(
                {
                    'segment': segment_index,
                    'segment_center': float((start + stop) / 2.0),
                    'peak_label': str(window['label']),
                    'peak_position': peak_position,
                    'peak_count': int(window_values.size),
                }
            )
    return pd.DataFrame(rows)


def plot_tof_segment_drift(
    dataframe: pd.DataFrame,
    *,
    windows: Sequence[dict] | Sequence[Sequence] | None = None,
    num_segments: int = 20,
    max_value: float | None = None,
) -> plt.Figure | None:
    """Plot TOF peak position drift before calibration."""
    drift = compute_tof_segment_drift(
        dataframe,
        windows=windows,
        num_segments=num_segments,
        max_value=max_value,
    )
    if drift.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.5))
    for peak_label, subset in drift.groupby('peak_label'):
        axes[0].plot(subset['segment_center'], subset['peak_position'], marker='o', linewidth=1.5, label=peak_label)
        axes[1].plot(subset['segment_center'], subset['peak_count'], marker='o', linewidth=1.5, label=peak_label)
    axes[0].set_title('TOF peak position drift')
    axes[0].set_xlabel('Event index')
    axes[0].set_ylabel('Peak position (ns)')
    axes[1].set_title('TOF peak counts by segment')
    axes[1].set_xlabel('Event index')
    axes[1].set_ylabel('Count')
    axes[1].legend(loc='upper right')
    fig.tight_layout()
    return fig


def plot_detector_dead_zone_and_neighbors(
    dataframe: pd.DataFrame,
    *,
    max_points: int = 60000,
) -> plt.Figure | None:
    """Plot occupancy, nearest-neighbor distance, and dead-zone diagnostics directly from detector hits."""
    if 'x_det (cm)' not in dataframe.columns or 'y_det (cm)' not in dataframe.columns:
        return None

    detector = dataframe[['x_det (cm)', 'y_det (cm)']].to_numpy(dtype=float)
    finite = np.isfinite(detector).all(axis=1)
    detector = detector[finite]
    if len(detector) < 2:
        return None
    if len(detector) > max_points:
        step = max(1, len(detector) // max_points)
        detector = detector[::step]

    tree = cKDTree(detector)
    distances, _ = tree.query(detector, k=2)
    nearest = distances[:, 1]
    radius = np.hypot(detector[:, 0], detector[:, 1])

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    axes[0, 0].hexbin(detector[:, 0], detector[:, 1], gridsize=55, cmap='viridis', mincnt=1)
    axes[0, 0].set_title('Raw detector occupancy')
    axes[0, 0].set_xlabel('x_det (cm)')
    axes[0, 0].set_ylabel('y_det (cm)')
    axes[0, 0].set_aspect('equal', adjustable='box')

    nn_bins = _compute_histogram_bins(nearest, max(np.median(nearest) / 8.0, 1e-4))
    axes[0, 1].hist(nearest, bins=nn_bins, color='#2563eb', alpha=0.65, histtype='stepfilled', log=True)
    axes[0, 1].set_title('Nearest-neighbor distances')
    axes[0, 1].set_xlabel('Distance (cm)')
    axes[0, 1].set_ylabel('Count')

    axes[1, 0].hexbin(
        detector[:, 0],
        detector[:, 1],
        C=nearest,
        reduce_C_function=np.mean,
        gridsize=45,
        cmap='magma',
        mincnt=1,
    )
    axes[1, 0].set_title('Dead-zone / sparse-region map')
    axes[1, 0].set_xlabel('x_det (cm)')
    axes[1, 0].set_ylabel('y_det (cm)')
    axes[1, 0].set_aspect('equal', adjustable='box')

    if np.min(radius) != np.max(radius):
        edges = np.linspace(np.min(radius), np.max(radius), 25)
        bins = np.digitize(radius, edges)
        centers = []
        medians = []
        for bin_index in range(1, len(edges)):
            mask = bins == bin_index
            if np.count_nonzero(mask) < 10:
                continue
            centers.append(float(np.median(radius[mask])))
            medians.append(float(np.median(nearest[mask])))
        axes[1, 1].plot(centers, medians, marker='o', linewidth=1.5, color='#dc2626')
    axes[1, 1].set_title('Median nearest neighbor vs radius')
    axes[1, 1].set_xlabel('Detector radius (cm)')
    axes[1, 1].set_ylabel('Median NN distance (cm)')

    fig.tight_layout()
    return fig


def compute_same_pulse_detector_separations(
    dataframe: pd.DataFrame,
    *,
    group_column: str = 'start_counter',
    only_in_detector: bool = True,
    dlts_values: Sequence[int] | None = None,
    max_group_size: int = 64,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Compute pairwise detector-coordinate separations within the same pulse/group."""
    required = {group_column, 'x_det (cm)', 'y_det (cm)'}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Same-pulse separation diagnostics need columns: {sorted(missing)}")

    frame = dataframe.copy()
    if only_in_detector and 'in_detector' in frame.columns:
        frame = frame[frame['in_detector']]
    if dlts_values is not None and 'dlts' in frame.columns:
        allowed = {int(value) for value in dlts_values}
        frame = frame[frame['dlts'].isin(sorted(allowed))]
    frame = frame[np.isfinite(frame['x_det (cm)']) & np.isfinite(frame['y_det (cm)'])]
    if frame.empty:
        empty = pd.DataFrame(columns=[group_column, 'group_size', 'dx', 'dy', 'dr'])
        return empty, {
            'num_groups': 0,
            'groups_with_pairs': 0,
            'pair_count': 0,
            'skipped_large_groups': 0,
        }

    rows: list[pd.DataFrame] = []
    groups_with_pairs = 0
    skipped_large_groups = 0
    grouped_items = list(frame.groupby(group_column, sort=False))
    grouped_iter = (
        tqdm(grouped_items, desc='Computing same-pulse separations', unit='group') if show_progress else grouped_items
    )
    for _, group in grouped_iter:
        group_size = int(len(group))
        if group_size < 2:
            continue
        if group_size > max_group_size:
            skipped_large_groups += 1
            continue

        coords = group[['x_det (cm)', 'y_det (cm)']].to_numpy(dtype=float)
        dx_matrix = np.abs(coords[:, 0][:, None] - coords[:, 0][None, :])
        dy_matrix = np.abs(coords[:, 1][:, None] - coords[:, 1][None, :])
        upper = np.triu_indices(group_size, k=1)
        dx = dx_matrix[upper]
        dy = dy_matrix[upper]
        dr = np.hypot(dx, dy)
        if dx.size == 0:
            continue
        groups_with_pairs += 1
        rows.append(
            pd.DataFrame(
                {
                    group_column: np.repeat(group[group_column].iloc[0], dx.size),
                    'group_size': np.repeat(group_size, dx.size),
                    'dx': dx,
                    'dy': dy,
                    'dr': dr,
                }
            )
        )

    if not rows:
        empty = pd.DataFrame(columns=[group_column, 'group_size', 'dx', 'dy', 'dr'])
        return empty, {
            'num_groups': int(len(grouped_items)),
            'groups_with_pairs': 0,
            'pair_count': 0,
            'skipped_large_groups': int(skipped_large_groups),
        }

    pair_table = pd.concat(rows, ignore_index=True)
    dx_nonzero = pair_table.loc[pair_table['dx'] > 0, 'dx'].to_numpy(dtype=float)
    dy_nonzero = pair_table.loc[pair_table['dy'] > 0, 'dy'].to_numpy(dtype=float)
    dr_nonzero = pair_table.loc[pair_table['dr'] > 0, 'dr'].to_numpy(dtype=float)
    summary: dict[str, float | int] = {
        'num_groups': int(len(grouped_items)),
        'groups_with_pairs': int(groups_with_pairs),
        'pair_count': int(len(pair_table)),
        'skipped_large_groups': int(skipped_large_groups),
        'min_dx': float(np.min(dx_nonzero)) if dx_nonzero.size else float('nan'),
        'min_dy': float(np.min(dy_nonzero)) if dy_nonzero.size else float('nan'),
        'min_dr': float(np.min(dr_nonzero)) if dr_nonzero.size else float('nan'),
        'median_dx': float(np.median(dx_nonzero)) if dx_nonzero.size else float('nan'),
        'median_dy': float(np.median(dy_nonzero)) if dy_nonzero.size else float('nan'),
        'median_dr': float(np.median(dr_nonzero)) if dr_nonzero.size else float('nan'),
    }
    return pair_table, summary


def plot_same_pulse_detector_separations(
    pair_table: pd.DataFrame,
    *,
    bin_size: float = 0.1,
    title_prefix: str = 'Same-pulse detector separations',
) -> plt.Figure | None:
    """Plot pairwise dx/dy/dr separations for hits recovered from the same pulse."""
    if pair_table.empty:
        return None
    dx = pair_table['dx'].to_numpy(dtype=float)
    dy = pair_table['dy'].to_numpy(dtype=float)
    dr = pair_table['dr'].to_numpy(dtype=float)
    dx = dx[np.isfinite(dx)]
    dy = dy[np.isfinite(dy)]
    dr = dr[np.isfinite(dr)]
    dx_nonzero = dx[dx > 0]
    dy_nonzero = dy[dy > 0]
    dr_nonzero = dr[dr > 0]
    if dx_nonzero.size == 0 and dy_nonzero.size == 0 and dr_nonzero.size == 0:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    if dx_nonzero.size:
        dx_bins = _compute_histogram_bins(dx_nonzero, bin_size)
        axes[0, 0].hist(dx_nonzero, bins=dx_bins, color='#2563eb', alpha=0.75, histtype='step', log=True)
    axes[0, 0].set_title(f'{title_prefix}: |dx|')
    axes[0, 0].set_xlabel('dx (cm)')
    axes[0, 0].set_ylabel('Count')

    if dy_nonzero.size:
        dy_bins = _compute_histogram_bins(dy_nonzero, bin_size)
        axes[0, 1].hist(dy_nonzero, bins=dy_bins, color='#dc2626', alpha=0.75, histtype='step', log=True)
    axes[0, 1].set_title(f'{title_prefix}: |dy|')
    axes[0, 1].set_xlabel('dy (cm)')
    axes[0, 1].set_ylabel('Count')

    positive_components = pair_table[(pair_table['dx'] > 0) & (pair_table['dy'] > 0)]
    if not positive_components.empty:
        dx_comp = positive_components['dx'].to_numpy(dtype=float)
        dy_comp = positive_components['dy'].to_numpy(dtype=float)
        x_bins = _compute_histogram_bins(dx_comp, bin_size)
        y_bins = _compute_histogram_bins(dy_comp, bin_size)
        histogram = axes[1, 0].hist2d(dx_comp, dy_comp, bins=[x_bins, y_bins], cmap='viridis')
        fig.colorbar(histogram[3], ax=axes[1, 0], label='Count')
    axes[1, 0].set_title(f'{title_prefix}: dx/dy map')
    axes[1, 0].set_xlabel('dx (cm)')
    axes[1, 0].set_ylabel('dy (cm)')

    if dr_nonzero.size:
        dr_bins = _compute_histogram_bins(dr_nonzero, bin_size)
        axes[1, 1].hist(dr_nonzero, bins=dr_bins, color='#059669', alpha=0.75, histtype='step', log=True)
    axes[1, 1].set_title(f'{title_prefix}: radial separation')
    axes[1, 1].set_xlabel('dr (cm)')
    axes[1, 1].set_ylabel('Count')

    fig.tight_layout()
    return fig


def _calculate_delta_p_and_multi(
    start_counter: np.ndarray, *, max_start_counter: int = 20000
) -> tuple[np.ndarray, np.ndarray]:
    delta_p = np.zeros(len(start_counter), dtype=np.uint32)
    multi = np.zeros(len(start_counter), dtype=np.uint32)
    if len(start_counter) == 0:
        return delta_p, multi

    previous_counter = int(start_counter[0])
    run_start = 0
    delta_p[0] = 0
    for index in range(1, len(start_counter)):
        current_counter = int(start_counter[index])
        step = current_counter - previous_counter
        if step < 0:
            step = (max_start_counter - previous_counter) + current_counter
        delta_p[index] = step

        if current_counter != previous_counter:
            multi[run_start:index] = index - run_start
            run_start = index
            previous_counter = current_counter

    multi[run_start:] = len(start_counter) - run_start
    return delta_p, multi


def summarize_processed_dataset(
    source: TabularSource,
    *,
    chunk_size: int = lazy_io.DEFAULT_CHUNK_SIZE,
) -> dict[str, float]:
    """Return compact diagnostics for any processed PyCCAPT-style dataset.

    Accepts either a ``pandas.DataFrame`` or a
    :class:`pyccapt.calibration.data_tools.lazy_io.LazyTable`. In the lazy case
    the per-column statistics (min/max/median) are computed by streaming over
    fixed-size chunks; median is reported as the median of per-chunk medians,
    which is a stable diagnostic estimate (within bin resolution) and avoids
    needing the whole column in RAM.
    """
    summary = {
        'num_rows': _row_count(source),
    }
    if summary['num_rows'] == 0:
        return summary

    numeric_columns = ['mc (Da)', 't (ns)', 'high_voltage (V)', 'x_det (cm)', 'y_det (cm)', 'multi', 'delta_p']
    for column in numeric_columns:
        if not _has_column(source, column):
            continue
        running_min = math.inf
        running_max = -math.inf
        per_chunk_medians: list[float] = []
        per_chunk_weights: list[int] = []
        for chunk in _iter_column_chunks(source, column, chunk_size=chunk_size):
            finite = chunk[np.isfinite(chunk)]
            if finite.size == 0:
                continue
            running_min = min(running_min, float(np.min(finite)))
            running_max = max(running_max, float(np.max(finite)))
            per_chunk_medians.append(float(np.median(finite)))
            per_chunk_weights.append(int(finite.size))
        if not math.isfinite(running_min):
            continue
        summary[f'{column}_min'] = running_min
        summary[f'{column}_max'] = running_max
        if _is_lazy(source):
            # Weighted median of the per-chunk medians: a robust streaming
            # estimate that converges to the true median as chunk_size grows.
            medians_arr = np.asarray(per_chunk_medians, dtype=float)
            weights_arr = np.asarray(per_chunk_weights, dtype=float)
            order = np.argsort(medians_arr)
            medians_arr = medians_arr[order]
            weights_arr = weights_arr[order]
            cumulative = np.cumsum(weights_arr)
            target = cumulative[-1] / 2.0
            idx = int(np.searchsorted(cumulative, target))
            summary[f'{column}_median'] = float(medians_arr[min(idx, len(medians_arr) - 1)])
        else:
            # Eager path: exact median over the full column.
            values = source[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            summary[f'{column}_median'] = float(np.median(finite))
    return summary


def plot_processed_dataset_overview(
    source: TabularSource,
    *,
    mc_max: float | None = 80.0,
    tof_max: float | None = 2000.0,
    bin_size: float = 0.1,
    title_prefix: str = 'Processed dataset',
    chunk_size: int = lazy_io.DEFAULT_CHUNK_SIZE,
    history_max_points: int = 100_000,
) -> plt.Figure:
    """Plot detector, TOF, mass/charge, and experiment-history diagnostics for a processed dataset.

    Accepts either a ``pandas.DataFrame`` or a
    :class:`pyccapt.calibration.data_tools.lazy_io.LazyTable`. In the lazy
    case the TOF and mass-spectrum histograms are accumulated chunk-by-chunk;
    the detector heatmap uses a streaming 2-D histogram; the experiment
    history line plot is downsampled to ``history_max_points`` evenly spaced
    samples so peak RAM stays bounded.
    """
    if _row_count(source) == 0:
        raise ValueError('Processed dataset is empty')

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))

    # --- TOF panel ---------------------------------------------------------
    if _has_column(source, 't (ns)'):
        tof_lo, tof_hi = _streaming_value_range(source, 't (ns)', upper_clip=tof_max, chunk_size=chunk_size)
        if tof_lo is not None and tof_hi > tof_lo:
            tof_edges = np.arange(tof_lo, tof_hi + bin_size, bin_size, dtype=float)
            if tof_edges.size >= 2:
                tof_counts = _streaming_filtered_histogram(
                    source,
                    't (ns)',
                    tof_edges,
                    upper_clip=tof_max,
                    chunk_size=chunk_size,
                )
                axes[0, 0].stairs(tof_counts, tof_edges, fill=True, color='#2563eb', alpha=0.65)
                axes[0, 0].set_yscale('log')
    axes[0, 0].set_title(f'{title_prefix}: TOF')
    axes[0, 0].set_xlabel('Time of flight [ns]')
    axes[0, 0].set_ylabel('Count')

    # --- Mass-spectrum panel -------------------------------------------------
    if _has_column(source, 'mc (Da)'):
        mc_lo, mc_hi = _streaming_value_range(source, 'mc (Da)', upper_clip=mc_max, chunk_size=chunk_size)
        if mc_lo is not None and mc_hi > mc_lo:
            mc_edges = np.arange(mc_lo, mc_hi + bin_size, bin_size, dtype=float)
            if mc_edges.size >= 2:
                mc_counts = _streaming_filtered_histogram(
                    source,
                    'mc (Da)',
                    mc_edges,
                    upper_clip=mc_max,
                    chunk_size=chunk_size,
                )
                axes[0, 1].stairs(mc_counts, mc_edges, fill=True, color='#d97706', alpha=0.65)
                axes[0, 1].set_yscale('log')
    axes[0, 1].set_title(f'{title_prefix}: Mass spectrum')
    axes[0, 1].set_xlabel('Mass-to-charge [Da]')
    axes[0, 1].set_ylabel('Count')

    # --- Detector heatmap ----------------------------------------------------
    if _has_column(source, 'x_det (cm)') and _has_column(source, 'y_det (cm)'):
        _plot_streaming_detector_map(
            axes[1, 0],
            source,
            chunk_size=chunk_size,
        )
    axes[1, 0].set_title(f'{title_prefix}: Detector map')
    axes[1, 0].set_xlabel('x_det (cm)')
    axes[1, 0].set_ylabel('y_det (cm)')
    axes[1, 0].set_aspect('equal', adjustable='box')

    # --- Experiment history -------------------------------------------------
    n_rows = _row_count(source)
    if _has_column(source, 'high_voltage (V)'):
        idx, hv = _downsample_column(source, 'high_voltage (V)', history_max_points, chunk_size=chunk_size)
        axes[1, 1].plot(idx, hv, color='#111827', linewidth=0.8)
        axes[1, 1].set_ylabel('High voltage (V)')
    if _has_column(source, 'multi'):
        idx_m, multi = _downsample_column(source, 'multi', history_max_points, chunk_size=chunk_size)
        finite_multi = np.isfinite(multi)
        if np.any(finite_multi):
            axes_multi = axes[1, 1].twinx()
            axes_multi.plot(
                idx_m[finite_multi],
                multi[finite_multi],
                color='#dc2626',
                linewidth=0.8,
                alpha=0.55,
            )
            axes_multi.set_ylabel('Multi')
    axes[1, 1].set_title(f'{title_prefix}: Experiment history')
    axes[1, 1].set_xlabel('Ion index')
    axes[1, 1].set_xlim(0, max(n_rows - 1, 1))

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streaming-plot helpers used by plot_processed_dataset_overview.
# ---------------------------------------------------------------------------


def _streaming_value_range(
    source: TabularSource,
    column: str,
    *,
    upper_clip: float | None,
    chunk_size: int,
) -> tuple[float | None, float | None]:
    """Compute (min, max) over a column in one streaming pass with optional clip."""
    running_min = math.inf
    running_max = -math.inf
    for chunk in _iter_column_chunks(source, column, chunk_size=chunk_size):
        finite = chunk[np.isfinite(chunk)]
        if upper_clip is not None:
            finite = finite[finite <= upper_clip]
        if finite.size == 0:
            continue
        running_min = min(running_min, float(np.min(finite)))
        running_max = max(running_max, float(np.max(finite)))
    if not math.isfinite(running_min):
        return None, None
    return running_min, running_max


def _streaming_filtered_histogram(
    source: TabularSource,
    column: str,
    edges: np.ndarray,
    *,
    upper_clip: float | None,
    chunk_size: int,
) -> np.ndarray:
    """Accumulate a histogram with the same NaN / upper-clip filter as eager hist."""
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    for chunk in _iter_column_chunks(source, column, chunk_size=chunk_size):
        chunk = chunk[np.isfinite(chunk)]
        if upper_clip is not None:
            chunk = chunk[chunk <= upper_clip]
        if chunk.size == 0:
            continue
        c_counts, _ = np.histogram(chunk, bins=edges)
        counts += c_counts.astype(np.int64, copy=False)
    return counts


def _plot_streaming_detector_map(ax, source: TabularSource, *, chunk_size: int) -> None:
    """Render the detector occupancy heatmap from a streaming 2-D histogram."""
    x_lo, x_hi = _streaming_value_range(source, 'x_det (cm)', upper_clip=None, chunk_size=chunk_size)
    y_lo, y_hi = _streaming_value_range(source, 'y_det (cm)', upper_clip=None, chunk_size=chunk_size)
    if x_lo is None or y_lo is None or x_hi <= x_lo or y_hi <= y_lo:
        return
    grid = 55
    x_edges = np.linspace(x_lo, x_hi, grid + 1)
    y_edges = np.linspace(y_lo, y_hi, grid + 1)
    counts = np.zeros((grid, grid), dtype=np.int64)
    for chunk in _iter_columns_chunks(source, ['x_det (cm)', 'y_det (cm)'], chunk_size=chunk_size):
        xs = np.asarray(chunk['x_det (cm)'], dtype=float)
        ys = np.asarray(chunk['y_det (cm)'], dtype=float)
        finite = np.isfinite(xs) & np.isfinite(ys)
        if not np.any(finite):
            continue
        hist, _, _ = np.histogram2d(xs[finite], ys[finite], bins=[x_edges, y_edges])
        counts += hist.astype(np.int64, copy=False)
    if counts.sum() == 0:
        return
    ax.pcolormesh(x_edges, y_edges, counts.T, cmap='viridis', shading='auto')


def _downsample_column(
    source: TabularSource,
    column: str,
    max_points: int,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(indices, values)`` for a column, evenly sampled to at most ``max_points``.

    Avoids ever materializing the full column into RAM by sampling stride-by-
    stride within each chunk.
    """
    n_rows = _row_count(source)
    if n_rows <= max_points:
        if _is_lazy(source):
            col = source[column]
            values = np.asarray(col[:n_rows], dtype=float)
        else:
            values = source[column].to_numpy(dtype=float)[:n_rows]
        return np.arange(n_rows, dtype=int), values

    stride = max(1, n_rows // max_points)
    sampled_idx: list[np.ndarray] = []
    sampled_val: list[np.ndarray] = []
    if _is_lazy(source):
        col = source[column]
        offset = 0
        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            chunk = np.asarray(col[start:stop], dtype=float)
            first_local = (-start) % stride
            if first_local >= chunk.size:
                offset += chunk.size
                continue
            local = np.arange(first_local, chunk.size, stride, dtype=int)
            sampled_idx.append(local + start)
            sampled_val.append(chunk[local])
            offset += chunk.size
    else:
        full = source[column].to_numpy(dtype=float)
        idx = np.arange(0, n_rows, stride, dtype=int)
        return idx, full[idx]
    if not sampled_idx:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    return np.concatenate(sampled_idx), np.concatenate(sampled_val)


def save_processed_raw_dataset(dataframe: pd.DataFrame, output_path: str) -> None:
    """Save a processed raw-data workflow result to HDF5.

    Uses pytables ``format='table'`` so that downstream loaders on low-memory
    machines can iterate the file via ``pd.read_hdf(..., iterator=True,
    chunksize=...)`` instead of materializing the whole frame.
    """
    data_tools.store_df_to_hdf(dataframe, 'df', output_path, format='table')
