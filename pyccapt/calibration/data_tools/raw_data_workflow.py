"""Reusable raw-data analysis helpers extracted from tutorial notebooks."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from pyccapt.calibration.data_tools import data_loadcrop, data_tools, raw_data_surface_concept
from pyccapt.calibration.mc import mc_tools

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


def parse_roentdek_events(file_path: str) -> list[dict]:
    """Parse a RoentDek processed text file into event dictionaries."""
    events = []
    current_event = None
    event_started = False

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            event_header_match = re.match(r'------- #(\d+) -------', line)
            if event_header_match:
                if current_event is not None:
                    events.append(current_event)
                current_event = {'event_number': int(event_header_match.group(1)), 'channels': []}
                event_started = True
                continue

            if event_started and current_event is not None:
                if line.startswith('T'):
                    t_match = re.search(r'T\s+=\s+([\d.]+)\s+ns', line)
                    if t_match:
                        current_event['T'] = float(t_match.group(1))
                elif line.startswith('dT'):
                    dt_match = re.search(r'dT\s+=\s+([\d.]+)\s+ns', line)
                    if dt_match:
                        current_event['dT'] = float(dt_match.group(1))

                channel_match = re.match(r'chan\s+(\d+)\s+(\d+)\s+(.*)', line)
                if channel_match:
                    channel_num = int(channel_match.group(1))
                    num_values = int(channel_match.group(2))
                    values_str = channel_match.group(3).strip()
                    values = [float(values_str)] if num_values == 1 else [float(val) for val in values_str.split()]
                    current_event['channels'].append(
                        {'channel': channel_num, 'num_values': num_values, 'values': values}
                    )

    if current_event is not None:
        events.append(current_event)
    return events


def _initialize_roentdek_counters() -> dict[str, dict[int, int]]:
    return {
        'total': {i: 0 for i in range(1, 500)},
        'invalid': {i: 0 for i in range(1, 500)},
        'dld2': {i: 0 for i in range(1, 500)},
        'dld4': {i: 0 for i in range(1, 500)},
        'dld6': {i: 0 for i in range(1, 500)},
    }


def _extract_max_dld_patterns(channel_counts: dict[int, int]) -> dict[str, int]:
    patterns, _ = _extract_roentdek_pattern_details(channel_counts)
    return patterns


def _extract_roentdek_pattern_details(channel_counts: dict[int, int]) -> tuple[dict[str, int], dict[int, int]]:
    patterns = {'dld2': 0, 'dld4': 0, 'dld6': 0}
    counts = channel_counts.copy()

    while all(counts.get(i, 0) >= 1 for i in range(1, 7)):
        patterns['dld6'] += 1
        for i in range(1, 7):
            counts[i] -= 1

    while True:
        if counts.get(1, 0) >= 1 and counts.get(2, 0) >= 1 and counts.get(3, 0) >= 1 and counts.get(4, 0) >= 1:
            for i in (1, 2, 3, 4):
                counts[i] -= 1
            patterns['dld4'] += 1
        elif counts.get(1, 0) >= 1 and counts.get(2, 0) >= 1 and counts.get(5, 0) >= 1 and counts.get(6, 0) >= 1:
            for i in (1, 2, 5, 6):
                counts[i] -= 1
            patterns['dld4'] += 1
        elif counts.get(3, 0) >= 1 and counts.get(4, 0) >= 1 and counts.get(5, 0) >= 1 and counts.get(6, 0) >= 1:
            for i in (3, 4, 5, 6):
                counts[i] -= 1
            patterns['dld4'] += 1
        else:
            break

    while True:
        if counts.get(1, 0) >= 1 and counts.get(2, 0) >= 1:
            counts[1] -= 1
            counts[2] -= 1
            patterns['dld2'] += 1
        elif counts.get(3, 0) >= 1 and counts.get(4, 0) >= 1:
            counts[3] -= 1
            counts[4] -= 1
            patterns['dld2'] += 1
        elif counts.get(5, 0) >= 1 and counts.get(6, 0) >= 1:
            counts[5] -= 1
            counts[6] -= 1
            patterns['dld2'] += 1
        else:
            break

    remaining = {channel: int(value) for channel, value in counts.items() if int(value) > 0}
    return patterns, remaining


def summarize_roentdek_raw_events(events: list[dict]) -> dict[str, object]:
    """Summarize raw RoentDek event quality before flattening numeric tables."""
    channel_timestamp_totals: Counter[int] = Counter()
    channel_event_totals: Counter[int] = Counter()
    event_size_counts: Counter[int] = Counter()
    unmatched_timestamps: Counter[int] = Counter()
    pair_missing_partner_events: Counter[str] = Counter()
    pair_unbalanced_events: Counter[str] = Counter()

    multi_hit_events = 0
    invalid_pattern_events = 0
    matched_pattern_events = 0
    unmatched_pattern_events = 0
    total_timestamps = 0

    pair_labels = ((1, 2, '1-2'), (3, 4, '3-4'), (5, 6, '5-6'))
    for event in events:
        channel_counts = {int(channel['channel']): int(channel['num_values']) for channel in event.get('channels', [])}
        total_values = sum(channel_counts.get(i, 0) for i in range(1, 7))
        total_timestamps += total_values
        event_size_counts[total_values] += 1

        for channel_index in range(1, 7):
            count = channel_counts.get(channel_index, 0)
            channel_timestamp_totals[channel_index] += count
            if count > 0:
                channel_event_totals[channel_index] += 1

        patterns, remaining = _extract_roentdek_pattern_details(channel_counts)
        recovered_patterns = int(sum(patterns.values()))
        if recovered_patterns > 0:
            matched_pattern_events += 1
        elif total_values > 0:
            invalid_pattern_events += 1

        if recovered_patterns > 1:
            multi_hit_events += 1

        if remaining:
            unmatched_pattern_events += 1
            unmatched_timestamps.update(remaining)

        for first_channel, second_channel, label in pair_labels:
            first_count = channel_counts.get(first_channel, 0)
            second_count = channel_counts.get(second_channel, 0)
            if bool(first_count) ^ bool(second_count):
                pair_missing_partner_events[label] += 1
            if first_count != second_count:
                pair_unbalanced_events[label] += 1

    return {
        'total_events': int(len(events)),
        'total_timestamps': int(total_timestamps),
        'event_size_counts': dict(event_size_counts),
        'channel_timestamp_totals': {channel: int(channel_timestamp_totals[channel]) for channel in range(1, 7)},
        'channel_event_totals': {channel: int(channel_event_totals[channel]) for channel in range(1, 7)},
        'matched_pattern_events': int(matched_pattern_events),
        'invalid_pattern_events': int(invalid_pattern_events),
        'unmatched_pattern_events': int(unmatched_pattern_events),
        'multi_hit_events': int(multi_hit_events),
        'unmatched_timestamps': {channel: int(unmatched_timestamps[channel]) for channel in range(1, 7)},
        'pair_missing_partner_events': dict(pair_missing_partner_events),
        'pair_unbalanced_events': dict(pair_unbalanced_events),
    }


def classify_roentdek_events(events: list[dict]) -> tuple[list[dict], dict[str, dict[int, int]]]:
    """Annotate RoentDek events with DLTS pattern counts and return counters."""
    counters = _initialize_roentdek_counters()
    for event in events:
        channel_counts = {channel['channel']: channel['num_values'] for channel in event['channels']}
        total_values = sum(channel_counts.get(i, 0) for i in range(1, 7))
        if total_values == 0:
            continue
        counters['total'][total_values] += 1
        patterns, remaining = _extract_roentdek_pattern_details(channel_counts)
        if sum(patterns.values()) == 0:
            counters['invalid'][total_values] += 1
        else:
            counters['dld2'][total_values] += patterns['dld2']
            counters['dld4'][total_values] += patterns['dld4']
            counters['dld6'][total_values] += patterns['dld6']
        event['dlts'] = [6] * patterns['dld6'] + [4] * patterns['dld4'] + [2] * patterns['dld2']
        event['unmatched_channels'] = remaining
    return events, counters


def attach_roentdek_measurements(
    events: list[dict],
    numeric_table: np.ndarray,
    *,
    detx_columns: Sequence[int] | str = DEFAULT_ROENTDEK_DETX_COLUMNS,
    dety_columns: Sequence[int] | str = DEFAULT_ROENTDEK_DETY_COLUMNS,
    signal_columns: Sequence[int] | str = DEFAULT_ROENTDEK_SIGNAL_COLUMNS,
    signal_kind: str = 'tof',
    drop_zero_signal: bool = True,
    high_voltage_column: int | None = None,
    pulse_column: int | None = None,
    start_counter_column: int | None = None,
) -> list[dict]:
    """Attach detector coordinates and signal values from a numeric text table."""
    signal_kind = _normalize_signal_kind(signal_kind)
    detx_positions = _normalize_index_positions(detx_columns, 'detx_columns')
    dety_positions = _normalize_index_positions(dety_columns, 'dety_columns')
    signal_positions = _normalize_index_positions(signal_columns, 'signal_columns')
    if not (len(detx_positions) == len(dety_positions) == len(signal_positions)):
        raise ValueError('detx_columns, dety_columns, and signal_columns must have the same length')

    numeric_table = np.atleast_2d(numeric_table)
    if len(events) != len(numeric_table):
        raise ValueError(
            f'RoentDek event text and numeric table length mismatch: {len(events)} events vs {len(numeric_table)} rows'
        )

    _validate_numeric_table_columns(numeric_table, detx_positions, 'detx_columns')
    _validate_numeric_table_columns(numeric_table, dety_positions, 'dety_columns')
    _validate_numeric_table_columns(numeric_table, signal_positions, 'signal_columns')
    optional_columns = [column for column in (high_voltage_column, pulse_column, start_counter_column) if column is not None]
    if optional_columns:
        _validate_numeric_table_columns(numeric_table, optional_columns, 'optional columns')

    enriched_events = []
    for event, row in zip(events, numeric_table):
        detx_values = []
        dety_values = []
        signal_values = []
        for detx_pos, dety_pos, signal_pos in zip(detx_positions, dety_positions, signal_positions):
            signal_value = float(row[signal_pos])
            if drop_zero_signal and math.isclose(signal_value, 0.0):
                continue
            detx_values.append(float(row[detx_pos]))
            dety_values.append(float(row[dety_pos]))
            signal_values.append(signal_value)

        if not signal_values:
            continue

        enriched_event = dict(event)
        enriched_event['detx'] = detx_values
        enriched_event['dety'] = dety_values
        enriched_event['signal'] = signal_values
        enriched_event['signal_kind'] = signal_kind
        if high_voltage_column is not None:
            enriched_event['high_voltage (V)'] = float(row[high_voltage_column])
        if pulse_column is not None:
            enriched_event['pulse'] = float(row[pulse_column])
        if start_counter_column is not None:
            enriched_event['start_counter'] = int(row[start_counter_column])
        enriched_events.append(enriched_event)

    return enriched_events


def roentdek_hits_to_dataframe(events: list[dict], *, signal_kind: str = 'tof') -> pd.DataFrame:
    """Flatten enriched RoentDek events to a hit-level dataframe."""
    signal_kind = _normalize_signal_kind(signal_kind)
    rows = []
    for event in events:
        signal_values = event.get('signal', [])
        if not signal_values:
            continue
        dlts_values = event.get('dlts', [])
        detx_values = event.get('detx', [])
        dety_values = event.get('dety', [])
        num_hits = min(len(signal_values), len(detx_values), len(dety_values))
        for hit_index in range(num_hits):
            row = {
                'event_number': int(event.get('event_number', -1)),
                'hit_index': hit_index,
                'dlts': int(dlts_values[hit_index]) if hit_index < len(dlts_values) else np.nan,
                'x_det (cm)': float(detx_values[hit_index]),
                'y_det (cm)': float(dety_values[hit_index]),
                'tof (ns)': np.nan,
                'mc (Da)': np.nan,
            }
            if signal_kind == 'tof':
                row['tof (ns)'] = float(signal_values[hit_index])
            else:
                row['mc (Da)'] = float(signal_values[hit_index])
            if 'high_voltage (V)' in event:
                row['high_voltage (V)'] = float(event['high_voltage (V)'])
            if 'pulse' in event:
                row['pulse'] = float(event['pulse'])
            if 'start_counter' in event:
                row['start_counter'] = int(event['start_counter'])
            rows.append(row)
    return pd.DataFrame(rows)


def analyze_roentdek_dataset(
    processed_text_path: str,
    *,
    numeric_values_path: str | None = None,
    detx_columns: Sequence[int] | str = DEFAULT_ROENTDEK_DETX_COLUMNS,
    dety_columns: Sequence[int] | str = DEFAULT_ROENTDEK_DETY_COLUMNS,
    signal_columns: Sequence[int] | str = DEFAULT_ROENTDEK_SIGNAL_COLUMNS,
    signal_kind: str = 'tof',
    drop_zero_signal: bool = True,
) -> dict:
    """Parse, classify, and optionally enrich a RoentDek raw-data analysis dataset."""
    events = parse_roentdek_events(processed_text_path)
    events, counters = classify_roentdek_events(events)
    raw_summary = summarize_roentdek_raw_events(events)

    numeric_table = None
    enriched_events = events
    hit_table = pd.DataFrame()
    if numeric_values_path:
        numeric_table = load_numeric_text_table(numeric_values_path)
        enriched_events = attach_roentdek_measurements(
            events,
            numeric_table,
            detx_columns=detx_columns,
            dety_columns=dety_columns,
            signal_columns=signal_columns,
            signal_kind=signal_kind,
            drop_zero_signal=drop_zero_signal,
        )
        hit_table = roentdek_hits_to_dataframe(enriched_events, signal_kind=signal_kind)

    return {
        'events': enriched_events,
        'counters': counters,
        'raw_summary': raw_summary,
        'hit_table': hit_table,
        'numeric_table': numeric_table,
        'signal_kind': _normalize_signal_kind(signal_kind),
    }


def plot_roentdek_statistics(counters: dict[str, dict[int, int]], max_bins: int = 20) -> plt.Figure:
    """Plot RoentDek delay-line statistics."""
    bins = np.arange(1, max_bins + 1)
    total_arr = np.array([counters['total'].get(i, 0) for i in bins])
    dld2_arr = np.array([counters['dld2'].get(i, 0) for i in bins])
    dld4_arr = np.array([counters['dld4'].get(i, 0) for i in bins])
    dld6_arr = np.array([counters['dld6'].get(i, 0) for i in bins])

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    width = 0.2
    ax.bar(bins, total_arr, width=width * 3, label='Frequency', alpha=0.35, color='#9ca3af')
    ax.bar(bins - width, dld2_arr, width=width, label='2 DLTS', color=DLTS_COLORS[2])
    ax.bar(bins, dld4_arr, width=width, label='4 DLTS', color=DLTS_COLORS[4])
    ax.bar(bins + width, dld6_arr, width=width, label='6 DLTS', color=DLTS_COLORS[6])
    ax.set_xlabel('Number of delay-line timestamps per pulse')
    ax.set_ylabel('Count')
    if np.any(total_arr > 0):
        ax.set_yscale('log')
    ax.set_xticks(bins)
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig


def _surface_concept_position_from_pair(first_time: float, second_time: float) -> float:
    difference = second_time - first_time
    shifted = -0.5 * difference + XY_BIN_SHIFT
    return ((shifted - XY_BIN_SHIFT) * XY_FACTOR) * 0.1


def _surface_concept_hit_from_time_data(time_data_chunk: Sequence[int] | np.ndarray) -> tuple[float, float, float]:
    time_data_tmp = np.asarray(time_data_chunk, dtype=np.int64)
    det_x = _surface_concept_position_from_pair(time_data_tmp[0], time_data_tmp[1])
    det_y = _surface_concept_position_from_pair(time_data_tmp[2], time_data_tmp[3])
    tof = float(np.sum(time_data_tmp)) * TOF_FACTOR_NS
    return det_x, det_y, tof


def _recover_surface_concept_partial_hits(chunk_channels: np.ndarray, chunk_times: np.ndarray) -> list[dict]:
    recovered_hits = []
    if len(chunk_channels) < 2:
        return recovered_hits

    pair_definitions = [
        ('x', 0, 1),
        ('y', 2, 3),
    ]
    for axis, first_channel, second_channel in pair_definitions:
        first_indices = list(np.where(chunk_channels == first_channel)[0])
        second_indices = list(np.where(chunk_channels == second_channel)[0])
        while first_indices and second_indices:
            first_index = int(first_indices.pop(0))
            second_index = int(second_indices.pop(0))
            first_time = float(chunk_times[first_index])
            second_time = float(chunk_times[second_index])
            position = _surface_concept_position_from_pair(first_time, second_time)
            tof = (first_time + second_time) * TOF_FACTOR_NS_1D
            if axis == 'x':
                recovered_hits.append(
                    {
                        'x_det (cm)': position,
                        'y_det (cm)': 0.0,
                        'tof (ns)': float(tof),
                        'detector_axis': 'x',
                    }
                )
            else:
                recovered_hits.append(
                    {
                        'x_det (cm)': 0.0,
                        'y_det (cm)': position,
                        'tof (ns)': float(tof),
                        'detector_axis': 'y',
                    }
                )
    return recovered_hits


def summarize_surface_concept_sequences(sequence_records: list[dict]) -> dict[str, dict[int, int]]:
    """Count sequence lengths and recoverable 2-D/1-D groups."""
    total_counts: Counter[int] = Counter()
    dld2_counts: Counter[int] = Counter()
    dld4_counts: Counter[int] = Counter()
    invalid_counts: Counter[int] = Counter()

    for record in sequence_records:
        channel_array = np.asarray(record.get('channels', []), dtype=np.int64)
        time_array = np.asarray(record.get('time_data', []), dtype=np.int64)
        length = int(len(channel_array))
        total_counts[length] += 1
        valid_events = list(record.get('valid_event', []))
        num_chunks = max(len(valid_events), math.ceil(length / 4))
        for chunk_index in range(num_chunks):
            start = chunk_index * 4
            stop = min(start + 4, length)
            if start >= stop:
                continue
            chunk_channels = channel_array[start:stop]
            chunk_times = time_array[start:stop]
            is_valid = chunk_index < len(valid_events) and bool(valid_events[chunk_index]) and len(chunk_channels) == 4
            if is_valid:
                dld4_counts[length] += 1
            else:
                partial_hits = _recover_surface_concept_partial_hits(chunk_channels, chunk_times)
                if partial_hits:
                    dld2_counts[length] += len(partial_hits)
                else:
                    invalid_counts[length] += 1

    return {
        'total': dict(total_counts),
        'dld2': dict(dld2_counts),
        'dld4': dict(dld4_counts),
        'invalid': dict(invalid_counts),
    }


def summarize_surface_concept_raw_sequences(sequence_records: list[dict]) -> dict[str, object]:
    """Return old-workflow-style Surface Concept raw statistics."""
    total_timestamps = 0
    channel_timestamp_totals: Counter[int] = Counter()

    valid_four_channel_groups = 0
    invalid_four_channel_groups = 0
    length_three_groups = 0
    length_two_groups = 0
    length_one_groups = 0
    multi_hit_groups_of_four = 0
    multi_hit_irregular = 0
    multi_hit_groups_of_four_timestamps = 0
    multi_hit_irregular_timestamps = 0

    for record in sequence_records:
        channel_array = np.asarray(record.get('channels', []), dtype=np.int64)
        length = int(len(channel_array))
        total_timestamps += length
        channel_timestamp_totals.update(int(channel) for channel in channel_array.tolist())

        if length == 4:
            if list(record.get('valid_event', [])) == [True]:
                valid_four_channel_groups += 1
            else:
                invalid_four_channel_groups += 1
        elif length == 3:
            length_three_groups += 1
        elif length == 2:
            length_two_groups += 1
        elif length == 1:
            length_one_groups += 1
        elif length > 4 and length % 4 == 0:
            multi_hit_groups_of_four += 1
            multi_hit_groups_of_four_timestamps += length
        elif length > 4:
            multi_hit_irregular += 1
            multi_hit_irregular_timestamps += length

    return {
        'total_sequences': int(len(sequence_records)),
        'total_timestamps': int(total_timestamps),
        'channel_timestamp_totals': {channel: int(channel_timestamp_totals[channel]) for channel in range(4)},
        'valid_four_channel_groups': int(valid_four_channel_groups),
        'invalid_four_channel_groups': int(invalid_four_channel_groups),
        'length_three_groups': int(length_three_groups),
        'length_two_groups': int(length_two_groups),
        'length_one_groups': int(length_one_groups),
        'multi_hit_groups_of_four': int(multi_hit_groups_of_four),
        'multi_hit_irregular': int(multi_hit_irregular),
        'multi_hit_groups_of_four_timestamps': int(multi_hit_groups_of_four_timestamps),
        'multi_hit_irregular_timestamps': int(multi_hit_irregular_timestamps),
    }


def plot_surface_concept_sequence_statistics(sequence_stats: dict[str, dict[int, int]], max_bins: int = 20) -> plt.Figure:
    """Plot Surface Concept delay-line statistics."""
    bins = np.arange(1, max_bins + 1)
    total_arr = np.array([sequence_stats['total'].get(i, 0) for i in bins])
    dld2_arr = np.array([sequence_stats['dld2'].get(i, 0) for i in bins])
    dld4_arr = np.array([sequence_stats['dld4'].get(i, 0) for i in bins])

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    width = 0.24
    ax.bar(bins, total_arr, width=width * 3, label='Frequency', alpha=0.35, color='#9ca3af')
    ax.bar(bins - 0.5 * width, dld2_arr, width=width, label='2 DLTS', color=DLTS_COLORS[2])
    ax.bar(bins + 0.5 * width, dld4_arr, width=width, label='4 DLTS', color=DLTS_COLORS[4])
    ax.set_xlabel('Number of delay-line timestamps per pulse')
    ax.set_ylabel('Count')
    if np.any(total_arr > 0):
        ax.set_yscale('log')
    ax.set_xticks(bins)
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig


def extract_surface_concept_hits(
    sequence_records: list[dict],
    *,
    detector_limit_cm: float = 4.0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Recover valid 4-DLTS hits and partial 2-DLTS hits from Surface Concept sequences."""
    diagnostics = build_surface_concept_recovery_diagnostics(sequence_records, detector_limit_cm=detector_limit_cm)
    recovered = diagnostics[diagnostics['dlts'] > 0].copy()
    hit_table = recovered.rename(columns={'accepted': 'in_detector'})
    hit_table['recovery'] = hit_table['dlts'].astype(int).astype(str) + ' DLTS'
    hit_table = hit_table[
        ['start_counter', 'high_voltage (V)', 'pulse', 'tof (ns)', 'x_det (cm)', 'y_det (cm)', 'dlts', 'detector_axis', 'recovery', 'in_detector']
    ].reset_index(drop=True)

    stats = {
        'recovered_hits': int(len(recovered)),
        'two_d_hits': int(np.count_nonzero(recovered['dlts'] == 4)),
        'one_d_hits': int(np.count_nonzero(recovered['dlts'] == 2)),
        'two_d_in_detector': int(np.count_nonzero((recovered['dlts'] == 4) & recovered['accepted'])),
        'one_d_in_detector': int(np.count_nonzero((recovered['dlts'] == 2) & recovered['accepted'])),
        'outside_detector_hits': int(np.count_nonzero((recovered['dlts'] > 0) & (~recovered['accepted']))),
        'unrecoverable_chunks': int(np.count_nonzero(diagnostics['status'] == 'unrecoverable')),
    }
    return hit_table, stats


def plot_surface_concept_recovery_summary(recovery_stats: dict[str, int]) -> plt.Figure:
    """Plot a compact summary of recovered and rejected Surface Concept hits."""
    labels = [
        '4 DLTS in detector',
        '2 DLTS in detector',
        'Outside detector',
        'Unrecoverable',
    ]
    values = [
        int(recovery_stats.get('two_d_in_detector', 0)),
        int(recovery_stats.get('one_d_in_detector', 0)),
        int(recovery_stats.get('outside_detector_hits', 0)),
        int(recovery_stats.get('unrecoverable_chunks', 0)),
    ]
    colors = [DLTS_COLORS[4], DLTS_COLORS[2], '#6b7280', '#dc2626']

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel('Count')
    if max(values or [0]) > 20:
        ax.set_yscale('log')
    ax.tick_params(axis='x', rotation=25)
    fig.tight_layout()
    return fig


def build_surface_concept_recovery_diagnostics(
    sequence_records: list[dict],
    *,
    detector_limit_cm: float = 4.0,
) -> pd.DataFrame:
    """Build a per-candidate recovery table for advanced Surface Concept diagnostics."""
    rows = []
    for sequence_index, record in enumerate(sequence_records):
        channel_array = np.asarray(record.get('channels', []), dtype=np.int64)
        time_array = np.asarray(record.get('time_data', []), dtype=np.int64)
        length = len(channel_array)
        valid_events = list(record.get('valid_event', []))
        num_chunks = max(len(valid_events), math.ceil(length / 4))
        start_counter = int(record['start_counter'][0]) if len(record.get('start_counter', [])) else 0
        high_voltage = float(record.get('high_voltage', 0.0))
        pulse = float(record.get('pulse', 0.0))

        for chunk_index in range(num_chunks):
            start = chunk_index * 4
            stop = min(start + 4, length)
            if start >= stop:
                continue
            chunk_channels = channel_array[start:stop]
            chunk_times = time_array[start:stop]
            is_valid = chunk_index < len(valid_events) and bool(valid_events[chunk_index]) and len(chunk_times) == 4
            if is_valid:
                det_x, det_y, tof = _surface_concept_hit_from_time_data(chunk_times)
                radius = float(np.hypot(det_x, det_y))
                in_detector = abs(det_x) <= detector_limit_cm and abs(det_y) <= detector_limit_cm
                rows.append(
                    {
                        'sequence_index': sequence_index,
                        'chunk_index': chunk_index,
                        'start_counter': start_counter,
                        'high_voltage (V)': high_voltage,
                        'pulse': pulse,
                        'tof (ns)': tof,
                        'x_det (cm)': det_x,
                        'y_det (cm)': det_y,
                        'radius_cm': radius,
                        'dlts': 4,
                        'detector_axis': 'xy',
                        'accepted': in_detector,
                        'status': '4 DLTS in detector' if in_detector else '4 DLTS outside detector',
                    }
                )
                continue

            partial_hits = _recover_surface_concept_partial_hits(chunk_channels, chunk_times)
            if not partial_hits:
                rows.append(
                    {
                        'sequence_index': sequence_index,
                        'chunk_index': chunk_index,
                        'start_counter': start_counter,
                        'high_voltage (V)': high_voltage,
                        'pulse': pulse,
                        'tof (ns)': np.nan,
                        'x_det (cm)': np.nan,
                        'y_det (cm)': np.nan,
                        'radius_cm': np.nan,
                        'dlts': 0,
                        'detector_axis': 'none',
                        'accepted': False,
                        'status': 'unrecoverable',
                    }
                )
                continue

            for partial_hit in partial_hits:
                det_x = float(partial_hit['x_det (cm)'])
                det_y = float(partial_hit['y_det (cm)'])
                in_detector = abs(det_x) <= detector_limit_cm and abs(det_y) <= detector_limit_cm
                rows.append(
                    {
                        'sequence_index': sequence_index,
                        'chunk_index': chunk_index,
                        'start_counter': start_counter,
                        'high_voltage (V)': high_voltage,
                        'pulse': pulse,
                        'tof (ns)': float(partial_hit['tof (ns)']),
                        'x_det (cm)': det_x,
                        'y_det (cm)': det_y,
                        'radius_cm': float(np.hypot(det_x, det_y)),
                        'dlts': 2,
                        'detector_axis': str(partial_hit['detector_axis']),
                        'accepted': in_detector,
                        'status': '2 DLTS in detector' if in_detector else '2 DLTS outside detector',
                    }
                )
    return pd.DataFrame(rows)


def analyze_surface_concept_dataset(
    hdf5_path: str,
    *,
    detector_limit_cm: float = 4.0,
    t0: float = 0.0,
    flight_path_length_mm: float = 110.0,
    pulse_mode: str = 'voltage',
) -> dict:
    """Recover and analyze Surface Concept raw TDC hits."""
    df_tdc = data_loadcrop.fetch_dataset_from_dld_grp(hdf5_path, extract_mode='tdc_sc')
    sequence_records = raw_data_surface_concept.find_consecutive_sequences(
        df_tdc['start_counter'].to_numpy(),
        df_tdc['channel'].to_numpy(),
        df_tdc['time_data'].to_numpy(),
        df_tdc['high_voltage (V)'].to_numpy(),
        df_tdc['pulse'].to_numpy(),
        print_stats=False,
    )
    sequence_stats = summarize_surface_concept_sequences(sequence_records)
    raw_summary = summarize_surface_concept_raw_sequences(sequence_records)
    recovery_diagnostics = build_surface_concept_recovery_diagnostics(
        sequence_records,
        detector_limit_cm=detector_limit_cm,
    )
    hit_table, recovery_stats = extract_surface_concept_hits(sequence_records, detector_limit_cm=detector_limit_cm)

    if not hit_table.empty:
        pulse_for_mc = hit_table['pulse'].to_numpy() if pulse_mode == 'voltage' else np.zeros(len(hit_table))
        hit_table['mc (Da)'] = mc_tools.tof2mc(
            t=hit_table['tof (ns)'].to_numpy(),
            t0=t0,
            V=hit_table['high_voltage (V)'].to_numpy(),
            xDet=hit_table['x_det (cm)'].to_numpy(),
            yDet=hit_table['y_det (cm)'].to_numpy(),
            flightPathLength=flight_path_length_mm,
            V_pulse=pulse_for_mc,
            mode=pulse_mode,
        )

    return {
        'tdc_frame': df_tdc,
        'sequence_records': sequence_records,
        'sequence_stats': sequence_stats,
        'raw_summary': raw_summary,
        'recovery_diagnostics': recovery_diagnostics,
        'hit_table': hit_table,
        'recovery_stats': recovery_stats,
        'pulse_mode': pulse_mode,
        'flight_path_length_mm': flight_path_length_mm,
        't0': t0,
    }


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


def plot_surface_concept_recovery_yield(
    recovery_diagnostics: pd.DataFrame,
    *,
    num_bins: int = 20,
) -> plt.Figure | None:
    """Plot recovery-yield/composition trends versus event index, voltage, and detector radius."""
    if recovery_diagnostics.empty:
        return None

    statuses_all = ['4 DLTS in detector', '2 DLTS in detector', '2 DLTS outside detector', 'unrecoverable']
    statuses_radius = ['4 DLTS in detector', '2 DLTS in detector', '2 DLTS outside detector']
    colors = {
        '4 DLTS in detector': DLTS_COLORS[4],
        '2 DLTS in detector': DLTS_COLORS[2],
        '2 DLTS outside detector': '#6b7280',
        'unrecoverable': '#dc2626',
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.4))
    event_summary = _binned_status_fraction(recovery_diagnostics, 'sequence_index', statuses_all, num_bins=num_bins)
    voltage_summary = _binned_status_fraction(recovery_diagnostics, 'high_voltage (V)', statuses_all, num_bins=num_bins)
    radius_summary = _binned_status_fraction(recovery_diagnostics, 'radius_cm', statuses_radius, num_bins=num_bins)

    for axis, summary, x_label, title, statuses in [
        (axes[0], event_summary, 'Event index', 'Recovery yield vs event index', statuses_all),
        (axes[1], voltage_summary, 'High voltage (V)', 'Recovery yield vs voltage', statuses_all),
        (axes[2], radius_summary, 'Detector radius (cm)', 'Recovery yield vs detector radius', statuses_radius),
    ]:
        if summary.empty:
            axis.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=axis.transAxes)
            axis.set_title(title)
            axis.set_xlabel(x_label)
            axis.set_ylabel('Fraction')
            continue
        for status in statuses:
            status_frame = summary[summary['status'] == status]
            axis.plot(
                status_frame['bin_center'].to_numpy(),
                status_frame['fraction'].to_numpy(),
                label=status,
                linewidth=1.8,
                color=colors[status],
            )
        axis.set_ylim(0.0, 1.05)
        axis.set_xlabel(x_label)
        axis.set_ylabel('Fraction')
        axis.set_title(title)
    axes[2].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    return fig


def plot_partial_hit_efficiency_maps(recovery_diagnostics: pd.DataFrame) -> plt.Figure | None:
    """Plot partial-hit recovery efficiency diagnostics for x and y channel pairs."""
    partial = recovery_diagnostics[recovery_diagnostics['dlts'] == 2].copy()
    if partial.empty:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))
    pair_labels = {'x': 'Channels 0-1', 'y': 'Channels 2-3'}
    accepted_fraction = []
    total_counts = []
    accepted_counts = []
    for axis_name in ['x', 'y']:
        subset = partial[partial['detector_axis'] == axis_name]
        total = len(subset)
        accepted = int(np.count_nonzero(subset['accepted']))
        total_counts.append(total)
        accepted_counts.append(accepted)
        accepted_fraction.append((accepted / total) if total else 0.0)

    axes[0, 0].bar([pair_labels['x'], pair_labels['y']], accepted_fraction, color=[DLTS_COLORS[2], '#6b7280'])
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].set_ylabel('Accepted fraction')
    axes[0, 0].set_title('Partial-hit pair efficiency')

    axes[0, 1].bar([pair_labels['x'], pair_labels['y']], total_counts, color='#d1d5db', label='Total')
    axes[0, 1].bar([pair_labels['x'], pair_labels['y']], accepted_counts, color=DLTS_COLORS[2], label='Accepted')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Partial-hit pair counts')
    axes[0, 1].legend(loc='upper right')

    for axis_plot, axis_name in zip([axes[1, 0], axes[1, 1]], ['x', 'y']):
        subset = partial[partial['detector_axis'] == axis_name]
        if subset.empty:
            axis_plot.text(0.5, 0.5, 'No data', ha='center', va='center', transform=axis_plot.transAxes)
        else:
            positions = subset['x_det (cm)'].to_numpy(dtype=float) if axis_name == 'x' else subset['y_det (cm)'].to_numpy(dtype=float)
            event_index = subset['sequence_index'].to_numpy(dtype=float)
            accepted = subset['accepted'].to_numpy(dtype=float)
            axis_plot.hexbin(
                event_index,
                positions,
                C=accepted,
                reduce_C_function=np.mean,
                gridsize=35,
                cmap='viridis',
                mincnt=1,
            )
        axis_plot.set_xlabel('Sequence index')
        axis_plot.set_ylabel('Recovered position (cm)')
        axis_plot.set_title(f'{pair_labels[axis_name]} efficiency map')

    fig.tight_layout()
    return fig


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


def surface_concept_hits_to_processed_dataframe(
    hit_table: pd.DataFrame,
    *,
    pulse_mode: str = 'voltage',
    max_start_counter: int = 20000,
) -> pd.DataFrame:
    """Convert recovered Surface Concept hits into a PyCCAPT-style processed dataframe."""
    frame = hit_table.copy()
    if 'in_detector' in frame.columns:
        frame = frame[frame['in_detector']]
    if frame.empty:
        raise ValueError('No in-detector Surface Concept hits are available to build a processed dataframe')
    if 'mc (Da)' not in frame.columns:
        raise ValueError("The hit table must contain 'mc (Da)' before converting to a processed dataframe")

    pulse_v = frame['pulse'].to_numpy() if pulse_mode == 'voltage' else np.zeros(len(frame))
    pulse_l = frame['pulse'].to_numpy() if pulse_mode == 'laser' else np.zeros(len(frame))
    delta_p, multi = _calculate_delta_p_and_multi(frame['start_counter'].to_numpy(), max_start_counter=max_start_counter)

    processed = pd.DataFrame(
        {
            'x (nm)': np.zeros(len(frame)),
            'y (nm)': np.zeros(len(frame)),
            'z (nm)': np.zeros(len(frame)),
            'mc (Da)': frame['mc (Da)'].to_numpy(),
            'mc_uc (Da)': frame['mc (Da)'].to_numpy().copy(),
            'high_voltage (V)': frame['high_voltage (V)'].to_numpy(),
            'pulse_v (V)': pulse_v,
            'pulse_l (pJ)': pulse_l,
            't (ns)': frame['tof (ns)'].to_numpy(),
            't_c (ns)': frame['tof (ns)'].to_numpy().copy(),
            'x_det (cm)': frame['x_det (cm)'].to_numpy(),
            'y_det (cm)': frame['y_det (cm)'].to_numpy(),
            'delta_p': delta_p,
            'multi': multi,
            'start_counter': frame['start_counter'].to_numpy(dtype=np.uint32),
        }
    )
    return processed


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


def reconstruct_surface_concept_dataset(
    hdf5_path: str,
    *,
    flight_path_length_mm: float = 110.0,
    pulse_mode: str = 'voltage',
    t0: float = 0.0,
    detector_limit_cm: float = 4.0,
) -> pd.DataFrame:
    """Reconstruct a processed PyCCAPT-style dataset from a Surface Concept raw HDF5 file."""
    analysis = analyze_surface_concept_dataset(
        hdf5_path,
        detector_limit_cm=detector_limit_cm,
        t0=t0,
        flight_path_length_mm=flight_path_length_mm,
        pulse_mode=pulse_mode,
    )
    return surface_concept_hits_to_processed_dataframe(analysis['hit_table'], pulse_mode=pulse_mode)


def save_processed_raw_dataset(dataframe: pd.DataFrame, output_path: str) -> None:
    """Save a processed raw-data workflow result to HDF5."""
    data_tools.store_df_to_hdf(dataframe, 'df', output_path)
