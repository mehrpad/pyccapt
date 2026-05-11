"""RoentDek (3-delay-line / hexanode) raw-data workflow helpers.

Internal sibling of :mod:`raw_data_workflow`. Public surface is re-exported
from there.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pyccapt.calibration.data_tools._raw_workflow_common import (
    DEFAULT_ROENTDEK_DETX_COLUMNS,
    DEFAULT_ROENTDEK_DETY_COLUMNS,
    DEFAULT_ROENTDEK_SIGNAL_COLUMNS,
    DLTS_COLORS,
    _normalize_index_positions,
    _normalize_signal_kind,
    _validate_numeric_table_columns,
    load_numeric_text_table,
)


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
                    current_event['channels'].append({'channel': channel_num, 'num_values': num_values, 'values': values})

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


def analyze_roentdek_tdc_frame(
    df_tdc: pd.DataFrame,
    *,
    show_progress: bool = False,
) -> dict:
    """Analyze an already-loaded RoentDek-style raw tdc frame."""
    required = {'start_counter', 'channel'}
    missing = required.difference(df_tdc.columns)
    if missing:
        raise ValueError(f"RoentDek tdc frame is missing required columns: {sorted(missing)}")

    start_counter = df_tdc['start_counter'].to_numpy()
    channels = df_tdc['channel'].to_numpy()
    min_channel = int(np.min(channels)) if len(channels) else 0
    max_channel = int(np.max(channels)) if len(channels) else 0
    zero_based_channels = min_channel >= 0 and max_channel <= 5
    run_starts = np.r_[0, np.where(np.diff(start_counter) != 0)[0] + 1, len(df_tdc)].astype(np.int64)

    event_rows = []
    run_indices = range(len(run_starts) - 1)
    if show_progress:
        run_indices = tqdm(run_indices, desc='Analyzing RoentDek groups', unit='group')

    for event_number, run_index in enumerate(run_indices, start=1):
        start = int(run_starts[run_index])
        stop = int(run_starts[run_index + 1])
        run_frame = df_tdc.iloc[start:stop]
        counts = run_frame['channel'].value_counts().to_dict()
        normalized_counts = {
            (int(channel) + 1 if zero_based_channels else int(channel)): int(count) for channel, count in counts.items()
        }
        event = {
            'event_number': event_number,
            'start_counter': int(run_frame['start_counter'].iloc[0]),
            'channels': [
                {
                    'channel': int(channel),
                    'num_values': int(normalized_counts.get(channel, 0)),
                    'values': [0.0] * int(normalized_counts.get(channel, 0)),
                }
                for channel in sorted(int(value) for value in normalized_counts)
            ],
        }
        if 'event_group_id' in run_frame.columns:
            event['event_group_id'] = int(run_frame['event_group_id'].iloc[0])
        if 'high_voltage (V)' in run_frame.columns:
            event['high_voltage (V)'] = float(run_frame['high_voltage (V)'].iloc[0])
        if 'pulse_v (V)' in run_frame.columns:
            event['pulse'] = float(run_frame['pulse_v (V)'].iloc[0])
        elif 'pulse' in run_frame.columns:
            event['pulse'] = float(run_frame['pulse'].iloc[0])
        event_rows.append(event)

    events, counters = classify_roentdek_events(event_rows)
    raw_summary = summarize_roentdek_raw_events(events)
    return {
        'events': events,
        'counters': counters,
        'raw_summary': raw_summary,
        'tdc_frame': df_tdc,
    }


def roentdek_processed_to_hit_table(
    dld_df: pd.DataFrame,
    roentdek_analysis: dict,
) -> pd.DataFrame:
    """Attach RoentDek DLTS classes from raw tdc groups onto processed dld rows."""
    if 'event_group_id' not in dld_df.columns:
        return pd.DataFrame()

    event_map = {
        int(event['event_group_id']): list(event.get('dlts', []))
        for event in roentdek_analysis.get('events', [])
        if 'event_group_id' in event
    }
    rows = []
    grouped = dld_df.groupby('event_group_id', sort=False)
    for event_group_id, group in grouped:
        dlts_values = event_map.get(int(event_group_id), [])
        group = group.reset_index(drop=True)
        for row_index, (_, row) in enumerate(group.iterrows()):
            dlts = int(dlts_values[row_index]) if row_index < len(dlts_values) else np.nan
            x_det = float(row['x_det (cm)']) if 'x_det (cm)' in row else np.nan
            y_det = float(row['y_det (cm)']) if 'y_det (cm)' in row else np.nan
            hit = {
                'event_group_id': int(event_group_id),
                'hit_index': row_index,
                'dlts': dlts,
                'x_det (cm)': x_det,
                'y_det (cm)': y_det,
                'tof (ns)': float(row['t (ns)']) if 't (ns)' in row else np.nan,
                'mc (Da)': float(row['mc (Da)']) if 'mc (Da)' in row else np.nan,
                'high_voltage (V)': float(row['high_voltage (V)']) if 'high_voltage (V)' in row else np.nan,
                'pulse': float(row['pulse_v (V)']) if 'pulse_v (V)' in row else 0.0,
                'start_counter': int(row['start_counter']) if 'start_counter' in row else 0,
                'in_detector': bool(np.isfinite(x_det) and np.isfinite(y_det)),
            }
            rows.append(hit)
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
