"""Surface Concept (2-delay-line) raw-data workflow helpers.

Internal sibling of :mod:`raw_data_workflow`. Public surface is re-exported
from there.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from tqdm.auto import tqdm

from pyccapt.calibration.data_tools import data_loadcrop, raw_data_surface_concept
from pyccapt.calibration.data_tools._raw_workflow_common import (
    DLTS_COLORS,
    TOF_FACTOR_NS,
    TOF_FACTOR_NS_1D,
    XY_BIN_SHIFT,
    XY_FACTOR,
    _binned_status_fraction,
    _calculate_delta_p_and_multi,
    _normalize_signal_kind,
    normalize_signal_windows,
    summarize_signal_windows,
)
from pyccapt.calibration.mc import mc_tools


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


def _surface_concept_pulse_column(tdc_frame: pd.DataFrame, pulse_mode: str) -> str:
    mode = str(pulse_mode).strip().lower()
    if mode == 'laser':
        candidates = ('pulse_l (pJ)', 'pulse')
    else:
        candidates = ('pulse_v (V)', 'pulse')
    for column in candidates:
        if column in tdc_frame.columns:
            return column
    raise ValueError(
        f"Surface Concept tdc frame is missing the pulse column required for pulse_mode={pulse_mode!r}. "
        f"Tried: {candidates}."
    )


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
    show_progress: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Recover valid 4-DLTS hits and partial 2-DLTS hits from Surface Concept sequences."""
    diagnostics = build_surface_concept_recovery_diagnostics(
        sequence_records,
        detector_limit_cm=detector_limit_cm,
        show_progress=show_progress,
    )
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
    show_progress: bool = False,
) -> pd.DataFrame:
    """Build a per-candidate recovery table for advanced Surface Concept diagnostics."""
    rows = []
    records_iter = enumerate(sequence_records)
    if show_progress:
        records_iter = enumerate(
            tqdm(sequence_records, desc='Recovering Surface Concept hits', unit='sequence')
        )
    for sequence_index, record in records_iter:
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


def analyze_surface_concept_tdc_frame(
    df_tdc: pd.DataFrame,
    *,
    detector_limit_cm: float = 4.0,
    t0: float = 0.0,
    flight_path_length_mm: float = 110.0,
    pulse_mode: str = 'voltage',
    show_progress: bool = False,
) -> dict:
    """Recover and analyze Surface Concept hits from an already-loaded raw tdc frame."""
    required = {'start_counter', 'channel', 'time_data', 'high_voltage (V)'}
    missing = required.difference(df_tdc.columns)
    if missing:
        raise ValueError(f"Surface Concept tdc frame is missing required columns: {sorted(missing)}")

    pulse_column = _surface_concept_pulse_column(df_tdc, pulse_mode)
    sequence_records = raw_data_surface_concept.find_consecutive_sequences(
        df_tdc['start_counter'].to_numpy(),
        df_tdc['channel'].to_numpy(),
        df_tdc['time_data'].to_numpy(),
        df_tdc['high_voltage (V)'].to_numpy(),
        df_tdc[pulse_column].to_numpy(),
        print_stats=False,
    )
    sequence_stats = summarize_surface_concept_sequences(sequence_records)
    raw_summary = summarize_surface_concept_raw_sequences(sequence_records)
    recovery_diagnostics = build_surface_concept_recovery_diagnostics(
        sequence_records,
        detector_limit_cm=detector_limit_cm,
        show_progress=show_progress,
    )
    hit_table, recovery_stats = extract_surface_concept_hits(
        sequence_records,
        detector_limit_cm=detector_limit_cm,
        show_progress=False,
    )

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
    return analyze_surface_concept_tdc_frame(
        df_tdc,
        detector_limit_cm=detector_limit_cm,
        t0=t0,
        flight_path_length_mm=flight_path_length_mm,
        pulse_mode=pulse_mode,
    )


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


def summarize_surface_concept_peak_windows(
    hit_table: pd.DataFrame,
    recovery_diagnostics: pd.DataFrame,
    windows: Sequence[dict] | Sequence[Sequence] | None,
    *,
    signal_kind: str = 'mc',
    only_in_detector: bool = True,
) -> dict[str, pd.DataFrame | int]:
    """Summarize user-defined peak windows across 2-DLTS and 4-DLTS recovery classes."""
    normalized_windows = normalize_signal_windows(windows)
    if not normalized_windows:
        empty = pd.DataFrame(columns=['label', 'two_dlts_count', 'four_dlts_count'])
        empty_ratios = pd.DataFrame(
            columns=['Peak', 'Two DLTS count', 'Four DLTS count', 'Two DLTS %', 'Four DLTS %', 'Two/Four DLTS']
        )
        empty_bars = pd.DataFrame(columns=['label', 'count', 'color'])
        return {
            'counts': empty,
            'ratios': empty_ratios,
            'bars': empty_bars,
            'outside_detector_count': 0,
            'unrecoverable_count': 0,
            'total_in_detector': 0,
        }

    signal_value = _normalize_signal_kind(signal_kind)
    summary = summarize_signal_windows(
        hit_table,
        normalized_windows,
        signal_kind=signal_value,
        only_in_detector=only_in_detector,
    )

    labels = [str(window['label']) for window in normalized_windows]
    noise_label = 'Noise'
    ordered_labels = [*labels, noise_label]
    if summary.empty:
        summary = pd.DataFrame({'label': ordered_labels, 'dlts': np.zeros(len(ordered_labels)), 'count': np.zeros(len(ordered_labels))})

    def _count_for(label: str, dlts: int) -> int:
        matches = summary[(summary['label'] == label) & (summary['dlts'] == dlts)]
        if matches.empty:
            return 0
        return int(matches['count'].sum())

    total_in_detector = int(
        np.count_nonzero(
            (hit_table['dlts'].isin([2, 4]).to_numpy()) &
            (hit_table['in_detector'].to_numpy() if 'in_detector' in hit_table.columns else np.ones(len(hit_table), dtype=bool))
        )
    )
    outside_detector_count = int(
        np.count_nonzero(recovery_diagnostics['status'].isin(['2 DLTS outside detector', '4 DLTS outside detector']))
    )
    unrecoverable_count = int(np.count_nonzero(recovery_diagnostics['status'] == 'unrecoverable'))

    count_rows = []
    ratio_rows = []
    bar_rows = []
    for label in labels:
        two_count = _count_for(label, 2)
        four_count = _count_for(label, 4)
        count_rows.append(
            {
                'label': label,
                'two_dlts_count': two_count,
                'four_dlts_count': four_count,
            }
        )
        ratio_rows.append(
            {
                'Peak': label,
                'Two DLTS count': two_count,
                'Four DLTS count': four_count,
                'Two DLTS %': (100.0 * two_count / total_in_detector) if total_in_detector else 0.0,
                'Four DLTS %': (100.0 * four_count / total_in_detector) if total_in_detector else 0.0,
                'Two/Four DLTS': (two_count / four_count) if four_count else np.nan,
            }
        )
        bar_rows.append({'label': f'{label} 4 DLTS', 'count': four_count, 'color': '#f59e0b'})

    noise_four = _count_for(noise_label, 4)
    bar_rows.append({'label': f'{noise_label} 4 DLTS', 'count': noise_four, 'color': '#f59e0b'})

    for label in labels:
        two_count = _count_for(label, 2)
        bar_rows.append({'label': f'{label} 2 DLTS', 'count': two_count, 'color': '#10b981'})

    noise_two = _count_for(noise_label, 2)
    bar_rows.append({'label': f'{noise_label} 2 DLTS', 'count': noise_two, 'color': '#10b981'})
    bar_rows.append({'label': 'Outside detector', 'count': outside_detector_count, 'color': '#ef4444'})
    bar_rows.append({'label': 'Unrecoverable', 'count': unrecoverable_count, 'color': '#ef4444'})

    return {
        'counts': pd.DataFrame(count_rows),
        'ratios': pd.DataFrame(ratio_rows),
        'bars': pd.DataFrame(bar_rows),
        'outside_detector_count': outside_detector_count,
        'unrecoverable_count': unrecoverable_count,
        'total_in_detector': total_in_detector,
    }


def plot_surface_concept_peak_breakdown(
    peak_summary: dict[str, pd.DataFrame | int],
    *,
    title: str = 'Surface Concept peak-window breakdown',
) -> plt.Figure | None:
    """Plot a peak-by-peak 2-DLTS / 4-DLTS bar chart plus rejected-event counts."""
    bars = peak_summary.get('bars')
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        return None

    fig, ax = plt.subplots(figsize=(max(7.5, len(bars) * 0.7), 4.0))
    ax.bar(bars['label'], bars['count'], color=bars['color'], edgecolor='#4b5563', linewidth=0.6)
    ax.set_ylabel('Counts')
    ax.set_title(title, fontsize=11, fontweight='semibold')
    if np.nanmax(bars['count'].to_numpy(dtype=float)) > 20:
        ax.set_yscale('log')
    ax.tick_params(axis='x', rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#e5e7eb', linewidth=0.8, alpha=0.7)
    fig.tight_layout()
    return fig


def plot_surface_concept_peak_ratio_table(
    ratio_table: pd.DataFrame,
    *,
    title: str = 'Two delay line (Surface Concept)',
) -> plt.Figure | None:
    """Render a compact percentage/ratio table for the user-defined peak windows."""
    if ratio_table.empty:
        return None

    formatted = ratio_table.copy()
    formatted['Two DLTS %'] = formatted['Two DLTS %'].map(lambda value: f'{value:.2f}%')
    formatted['Four DLTS %'] = formatted['Four DLTS %'].map(lambda value: f'{value:.2f}%')
    formatted['Two/Four DLTS'] = formatted['Two/Four DLTS'].map(
        lambda value: 'n/a' if not np.isfinite(value) else f'{value:.3f}'
    )

    figure_width = max(6.8, 0.9 + 1.25 * len(formatted.columns))
    figure_height = max(2.6, 1.55 + 0.62 * len(formatted))
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    ax.axis('off')
    title_band_height = 0.17
    ax.add_patch(
        Rectangle(
            (0.0, 1.0 - title_band_height),
            1.0,
            title_band_height,
            transform=ax.transAxes,
            facecolor='#f3f4f6',
            edgecolor='#b6b8bb',
            linewidth=0.8,
        )
    )
    ax.text(
        0.03,
        1.0 - title_band_height / 2.0,
        title,
        transform=ax.transAxes,
        ha='left',
        va='center',
        fontsize=11,
        fontweight='semibold',
        color='black',
    )
    table = ax.table(
        cellText=formatted[['Peak', 'Two DLTS %', 'Four DLTS %', 'Two/Four DLTS']].to_numpy(),
        colLabels=['Ion', 'Two DLTS', 'Four DLTS', 'Two/four DLTS'],
        cellLoc='center',
        loc='center',
        colWidths=[0.17, 0.23, 0.23, 0.27],
        bbox=[0.0, 0.0, 1.0, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.35)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#b6b8bb')
        cell.set_linewidth(0.75)
        if row == 0:
            cell.set_facecolor('#f7f7f7')
            cell.set_text_props(weight='semibold', color='black')
            cell.set_height(cell.get_height() * 1.08)
        else:
            cell.set_facecolor('white')
            if col == 0:
                cell.set_text_props(ha='left')

    fig.tight_layout()
    return fig


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
