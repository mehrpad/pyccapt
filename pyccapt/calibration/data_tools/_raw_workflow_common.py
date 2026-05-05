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

from pyccapt.calibration.data_tools import data_tools

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
        raise ValueError(
            f'{label} requires column index {max_index}, but the numeric table only has {table.shape[1]} columns'
        )


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


def _auto_peak_windows(values: np.ndarray, *, max_value: float | None = None, max_peaks: int = 3) -> list[dict[str, float | str]]:
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
    dataframe: pd.DataFrame,
    *,
    windows: Sequence[dict] | Sequence[Sequence] | None = None,
    num_segments: int = 20,
    max_value: float | None = None,
) -> pd.DataFrame:
    """Estimate TOF peak drift by splitting the dataset into time/index segments."""
    tof_column = 'tof (ns)' if 'tof (ns)' in dataframe.columns else 't (ns)' if 't (ns)' in dataframe.columns else None
    if tof_column is None:
        raise ValueError("A TOF column ('tof (ns)' or 't (ns)') is required for drift analysis")

    values = dataframe[tof_column].to_numpy(dtype=float)
    normalized_windows = normalize_signal_windows(windows) if windows else _auto_peak_windows(values, max_value=max_value)
    if not normalized_windows:
        return pd.DataFrame(columns=['segment', 'peak_label', 'peak_position', 'peak_count'])

    segment_edges = np.linspace(0, len(values), num_segments + 1, dtype=int)
    rows = []
    for segment_index in range(num_segments):
        start = segment_edges[segment_index]
        stop = segment_edges[segment_index + 1]
        if stop <= start:
            continue
        segment_values = values[start:stop]
        segment_values = segment_values[np.isfinite(segment_values)]
        if max_value is not None:
            segment_values = segment_values[segment_values <= max_value]
        if segment_values.size == 0:
            continue
        for window in normalized_windows:
            window_values = segment_values[
                (segment_values >= float(window['min'])) &
                (segment_values <= float(window['max']))
            ]
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
    grouped_iter = tqdm(grouped_items, desc='Computing same-pulse separations', unit='group') if show_progress else grouped_items
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


def _calculate_delta_p_and_multi(start_counter: np.ndarray, *, max_start_counter: int = 20000) -> tuple[np.ndarray, np.ndarray]:
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


def summarize_processed_dataset(dataframe: pd.DataFrame) -> dict[str, float]:
    """Return compact diagnostics for any processed PyCCAPT-style dataset."""
    frame = dataframe.copy()
    summary = {
        'num_rows': int(len(frame)),
    }
    if len(frame) == 0:
        return summary

    numeric_columns = ['mc (Da)', 't (ns)', 'high_voltage (V)', 'x_det (cm)', 'y_det (cm)', 'multi', 'delta_p']
    for column in numeric_columns:
        if column not in frame.columns:
            continue
        values = frame[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        summary[f'{column}_min'] = float(np.min(finite))
        summary[f'{column}_max'] = float(np.max(finite))
        summary[f'{column}_median'] = float(np.median(finite))
    return summary


def plot_processed_dataset_overview(
    dataframe: pd.DataFrame,
    *,
    mc_max: float | None = 80.0,
    tof_max: float | None = 2000.0,
    bin_size: float = 0.1,
    title_prefix: str = 'Processed dataset',
) -> plt.Figure:
    """Plot detector, TOF, mass/charge, and experiment-history diagnostics for a processed dataset."""
    if dataframe.empty:
        raise ValueError('Processed dataset is empty')

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))

    tof = dataframe['t (ns)'].to_numpy(dtype=float) if 't (ns)' in dataframe.columns else np.array([])
    tof = tof[np.isfinite(tof)]
    if tof_max is not None:
        tof = tof[tof <= tof_max]
    if tof.size:
        tof_bins = _compute_histogram_bins(tof, bin_size)
        axes[0, 0].hist(tof, bins=tof_bins, color='#2563eb', alpha=0.65, log=True, histtype='stepfilled')
    axes[0, 0].set_title(f'{title_prefix}: TOF')
    axes[0, 0].set_xlabel('Time of flight [ns]')
    axes[0, 0].set_ylabel('Count')

    mc = dataframe['mc (Da)'].to_numpy(dtype=float) if 'mc (Da)' in dataframe.columns else np.array([])
    mc = mc[np.isfinite(mc)]
    if mc_max is not None:
        mc = mc[mc <= mc_max]
    if mc.size:
        mc_bins = _compute_histogram_bins(mc, bin_size)
        axes[0, 1].hist(mc, bins=mc_bins, color='#d97706', alpha=0.65, log=True, histtype='stepfilled')
    axes[0, 1].set_title(f'{title_prefix}: Mass spectrum')
    axes[0, 1].set_xlabel('Mass-to-charge [Da]')
    axes[0, 1].set_ylabel('Count')

    if 'x_det (cm)' in dataframe.columns and 'y_det (cm)' in dataframe.columns:
        x_det = dataframe['x_det (cm)'].to_numpy(dtype=float)
        y_det = dataframe['y_det (cm)'].to_numpy(dtype=float)
        finite = np.isfinite(x_det) & np.isfinite(y_det)
        if np.count_nonzero(finite) > 1200:
            axes[1, 0].hexbin(x_det[finite], y_det[finite], gridsize=55, cmap='viridis', mincnt=1)
        elif np.count_nonzero(finite):
            axes[1, 0].scatter(x_det[finite], y_det[finite], s=4, alpha=0.4, edgecolors='none', color='#059669')
    axes[1, 0].set_title(f'{title_prefix}: Detector map')
    axes[1, 0].set_xlabel('x_det (cm)')
    axes[1, 0].set_ylabel('y_det (cm)')
    axes[1, 0].set_aspect('equal', adjustable='box')

    x_axis = np.arange(len(dataframe))
    if 'high_voltage (V)' in dataframe.columns:
        axes[1, 1].plot(x_axis, dataframe['high_voltage (V)'].to_numpy(dtype=float), color='#111827', linewidth=0.8)
        axes[1, 1].set_ylabel('High voltage (V)')
    if 'multi' in dataframe.columns:
        multi = dataframe['multi'].to_numpy(dtype=float)
        finite_multi = np.isfinite(multi)
        if np.any(finite_multi):
            axes_multi = axes[1, 1].twinx()
            axes_multi.plot(
                x_axis[finite_multi],
                multi[finite_multi],
                color='#dc2626',
                linewidth=0.8,
                alpha=0.55,
            )
            axes_multi.set_ylabel('Multi')
    axes[1, 1].set_title(f'{title_prefix}: Experiment history')
    axes[1, 1].set_xlabel('Ion index')

    fig.tight_layout()
    return fig


def save_processed_raw_dataset(dataframe: pd.DataFrame, output_path: str) -> None:
    """Save a processed raw-data workflow result to HDF5."""
    data_tools.store_df_to_hdf(dataframe, 'df', output_path)
