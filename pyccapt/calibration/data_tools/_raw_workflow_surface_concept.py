"""Surface Concept (2-delay-line) raw-data workflow helpers.

Internal sibling of :mod:`raw_data_workflow`. Public surface is re-exported
from there.
"""

from __future__ import annotations

import gc
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


def _enumerate_axis_pairs(
    chunk_channels: np.ndarray,
    chunk_times: np.ndarray,
    first_channel: int,
    second_channel: int,
    detector_radius_cm: float,
    max_tof_ns: float | None,
) -> list[dict]:
    """Enumerate every physically-valid (first, second) pair for one axis.

    For a multi-hit pulse with n_A ticks on ``first_channel`` and n_B
    ticks on ``second_channel`` this returns up to n_A * n_B candidates,
    pre-filtered against the detector-surface and ToF gates. The caller
    is responsible for resolving ambiguity via bipartite matching so
    each tick contributes to at most one accepted hit.
    """
    first_indices = np.where(chunk_channels == first_channel)[0]
    second_indices = np.where(chunk_channels == second_channel)[0]
    if first_indices.size == 0 or second_indices.size == 0:
        return []

    candidates: list[dict] = []
    for i_local, i_global in enumerate(first_indices):
        t_first = float(chunk_times[int(i_global)])
        for j_local, j_global in enumerate(second_indices):
            t_second = float(chunk_times[int(j_global)])
            position = _surface_concept_position_from_pair(t_first, t_second)
            if abs(position) > detector_radius_cm:
                continue
            tof = (t_first + t_second) * TOF_FACTOR_NS_1D
            if max_tof_ns is not None and not (0.0 < tof <= max_tof_ns):
                continue
            candidates.append(
                {
                    'first_local': int(i_local),
                    'second_local': int(j_local),
                    'first_global': int(i_global),
                    'second_global': int(j_global),
                    'position': float(position),
                    'tof': float(tof),
                }
            )
    return candidates


def _select_axis_assignment(
    candidates: list[dict],
    n_first: int,
    n_second: int,
) -> list[dict]:
    """Pick a maximum-cardinality 1-to-1 assignment of candidate pairs.

    Bipartite matching ensures each ``first_channel`` tick and each
    ``second_channel`` tick participates in at most one selected pair —
    the physical "each timestamp belongs to exactly one ion" constraint.
    Among possible maximum matchings, the lowest |position| is preferred
    so the more central (and more likely physical) ions are chosen first.

    Implemented with ``scipy.optimize.linear_sum_assignment`` over a
    padded cost matrix: cost = |position| for valid pairs, +inf for
    invalid (so they are never selected unless forced, which is then
    rejected by the post-filter).
    """
    if not candidates:
        return []

    # Build a (n_first x n_second) cost matrix; missing pairs get +inf.
    cost = np.full((n_first, n_second), np.inf, dtype=np.float64)
    by_position: dict[tuple[int, int], dict] = {}
    for cand in candidates:
        key = (cand['first_local'], cand['second_local'])
        # Prefer the smaller-|position| candidate if duplicates ever
        # arise (they shouldn't from _enumerate_axis_pairs, but be safe).
        if key in by_position and abs(by_position[key]['position']) <= abs(cand['position']):
            continue
        by_position[key] = cand
        cost[cand['first_local'], cand['second_local']] = abs(cand['position'])

    # Pad to a square matrix so linear_sum_assignment is happy on
    # rectangular inputs. The pad rows/cols carry a finite large cost
    # which is still beaten by any finite real candidate, so they only
    # fill in when a real candidate cannot be matched.
    n_dim = max(n_first, n_second)
    if n_dim == 0:
        return []
    PAD_COST = 1e6  # well above any |position| <= 4 cm
    padded = np.full((n_dim, n_dim), PAD_COST, dtype=np.float64)
    padded[:n_first, :n_second] = cost
    # Replace inf with a large but finite value so the solver runs.
    padded[np.isinf(padded)] = PAD_COST

    from scipy.optimize import linear_sum_assignment

    row_ind, col_ind = linear_sum_assignment(padded)
    selected: list[dict] = []
    for r, c in zip(row_ind, col_ind):
        if r >= n_first or c >= n_second:
            continue  # padded slot; not a real pair
        cand = by_position.get((int(r), int(c)))
        if cand is None:
            continue  # no real candidate at this slot; padding picked it
        selected.append(cand)
    return selected


def _combine_axes_into_xy_hits(
    x_selected: list[dict],
    y_selected: list[dict],
    axis_consistency_ns: float | None,
    max_tof_ns: float | None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Cross-match accepted X-pairs and Y-pairs into full xy hits.

    Two ToFs from the same physical ion (one from each axis) must agree
    within ``axis_consistency_ns``. Pairs that agree are promoted to
    full 4-DLTS xy hits; those that do not are returned as leftover
    single-axis partials.

    Returns ``(xy_hits, x_leftover, y_leftover)``.
    """
    if not x_selected or not y_selected:
        return [], list(x_selected), list(y_selected)

    n_x, n_y = len(x_selected), len(y_selected)
    # Cost = |tof_x - tof_y|. A large pad keeps non-matches from being
    # forcibly selected.
    PAD_COST = 1e9
    n_dim = max(n_x, n_y)
    cost = np.full((n_dim, n_dim), PAD_COST, dtype=np.float64)
    for i, x_hit in enumerate(x_selected):
        for j, y_hit in enumerate(y_selected):
            diff = abs(x_hit['tof'] - y_hit['tof'])
            cost[i, j] = diff

    from scipy.optimize import linear_sum_assignment

    row_ind, col_ind = linear_sum_assignment(cost)

    used_x: set[int] = set()
    used_y: set[int] = set()
    xy_hits: list[dict] = []
    for i, j in zip(row_ind, col_ind):
        if i >= n_x or j >= n_y:
            continue
        x_hit = x_selected[int(i)]
        y_hit = y_selected[int(j)]
        diff = abs(x_hit['tof'] - y_hit['tof'])
        if axis_consistency_ns is not None and diff > axis_consistency_ns:
            continue
        combined_tof = 0.5 * (x_hit['tof'] + y_hit['tof'])
        if max_tof_ns is not None and not (0.0 < combined_tof <= max_tof_ns):
            continue
        xy_hits.append(
            {
                'x_det (cm)': x_hit['position'],
                'y_det (cm)': y_hit['position'],
                'tof (ns)': float(combined_tof),
                'detector_axis': 'xy',
                'tof_axis_x_ns': x_hit['tof'],
                'tof_axis_y_ns': y_hit['tof'],
            }
        )
        used_x.add(int(i))
        used_y.add(int(j))

    x_leftover = [hit for k, hit in enumerate(x_selected) if k not in used_x]
    y_leftover = [hit for k, hit in enumerate(y_selected) if k not in used_y]
    return xy_hits, x_leftover, y_leftover


def _recover_surface_concept_partial_hits(
    chunk_channels: np.ndarray,
    chunk_times: np.ndarray,
    *,
    detector_radius_cm: float = 4.0,
    max_tof_ns: float | None = None,
    axis_consistency_ns: float | None = 5.0,
    combine_axes: bool = True,
) -> list[dict]:
    """Recover every physically-valid hit from a multi-hit pulse.

    ``combine_axes`` (default True) cross-matches accepted x-pairs and
    y-pairs into full 4-DLTS xy hits -- the right behaviour for the live
    partial-recovery merge path, which produces FINAL hits. The
    diagnostics / reporting path (``build_surface_concept_recovery_diagnostics``)
    passes ``combine_axes=False`` together with ``detector_radius_cm=inf``
    so it can ENUMERATE every per-axis candidate pair and let its own
    ``detector_limit_cm`` check flag each hit's ``in_detector`` status,
    rather than silently dropping out-of-detector reconstructions or
    merging the two delay-line axes (which would under-count per-axis
    recoverability and mislabel dlts).

    For each delay-line axis (x: ch0+ch1, y: ch2+ch3) ALL pairwise
    combinations of one ch-A timestamp with one ch-B timestamp are
    enumerated, filtered against the detector-surface and ToF gates,
    and a maximum-cardinality 1-to-1 bipartite matching selects a
    consistent subset (each tick used at most once). Across axes, the
    selected x-pairs and y-pairs are then cross-matched by ToF
    agreement: a pair agreeing to within ``axis_consistency_ns``
    becomes a full 4-DLTS xy hit; the remaining unmatched pairs become
    2-DLTS single-axis partial hits.

    A pulse with 5 ch0, 5 ch1, 4 ch2, 4 ch3 ticks can therefore yield
    e.g. 3 full xy hits + 2 x-only partials + 1 y-only partial, all
    physically consistent and emitted as separate dicts.

    Parameters
    ----------
    chunk_channels : np.ndarray
        Per-tick channel ids in arrival order. Channel 0/1 form the
        x axis; channel 2/3 form the y axis.
    chunk_times : np.ndarray
        Per-tick raw TDC timestamps in arrival order.
    detector_radius_cm : float, default 4.0
        Pairs reconstructing |position| outside this radius are
        rejected as unphysical.
    max_tof_ns : float, optional
        If given, candidate ToFs outside ``(0, max_tof_ns]`` are
        rejected.
    axis_consistency_ns : float or None, default 5.0
        Maximum ``|tof_x - tof_y|`` permitted when promoting an
        (x-pair, y-pair) combination to a full xy hit. Set to ``None``
        to skip this check (any matched pair becomes full xy).
    """
    recovered_hits: list[dict] = []
    if chunk_channels is None or chunk_times is None or len(chunk_channels) < 2:
        return recovered_hits

    chunk_channels = np.asarray(chunk_channels, dtype=np.int64)
    chunk_times = np.asarray(chunk_times, dtype=np.int64)

    # --- Per-axis enumeration + 1-to-1 matching --------------------------
    pair_definitions = [
        ('x', 0, 1),
        ('y', 2, 3),
    ]
    per_axis_selected: dict[str, list[dict]] = {'x': [], 'y': []}
    for axis, first_channel, second_channel in pair_definitions:
        candidates = _enumerate_axis_pairs(
            chunk_channels,
            chunk_times,
            first_channel=first_channel,
            second_channel=second_channel,
            detector_radius_cm=detector_radius_cm,
            max_tof_ns=max_tof_ns,
        )
        if not candidates:
            continue
        n_first = int((chunk_channels == first_channel).sum())
        n_second = int((chunk_channels == second_channel).sum())
        per_axis_selected[axis] = _select_axis_assignment(candidates, n_first, n_second)

    # --- Cross-axis ToF agreement → promote to full xy hits --------------
    if combine_axes:
        xy_hits, x_leftover, y_leftover = _combine_axes_into_xy_hits(
            per_axis_selected['x'],
            per_axis_selected['y'],
            axis_consistency_ns=axis_consistency_ns,
            max_tof_ns=max_tof_ns,
        )
    else:
        # Diagnostics/reporting mode: keep the two delay-line axes
        # separate so every reconstructible per-axis pair is emitted as
        # its own 2-DLTS partial hit.
        xy_hits = []
        x_leftover = list(per_axis_selected['x'])
        y_leftover = list(per_axis_selected['y'])

    # --- Emit results ----------------------------------------------------
    # Full xy hits first (more physically informative), then leftover
    # single-axis partials. Each dict matches the schema the caller in
    # partial_recovery.py expects: x_det / y_det are NaN on the axis
    # that was not recovered.
    recovered_hits.extend(xy_hits)
    for x_hit in x_leftover:
        recovered_hits.append(
            {
                'x_det (cm)': x_hit['position'],
                'y_det (cm)': float('nan'),
                'tof (ns)': float(x_hit['tof']),
                'detector_axis': 'x',
                'tof_axis_x_ns': float(x_hit['tof']),
                'tof_axis_y_ns': float('nan'),
            }
        )
    for y_hit in y_leftover:
        recovered_hits.append(
            {
                'x_det (cm)': float('nan'),
                'y_det (cm)': y_hit['position'],
                'tof (ns)': float(y_hit['tof']),
                'detector_axis': 'y',
                'tof_axis_x_ns': float('nan'),
                'tof_axis_y_ns': float(y_hit['tof']),
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
        f"Surface Concept tdc frame is missing the pulse column required for pulse_mode={pulse_mode!r}. Tried: {candidates}."
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
    keep_columns = [
        'start_counter',
        'high_voltage (V)',
        'pulse',
        'tof (ns)',
        'x_det (cm)',
        'y_det (cm)',
        'dlts',
        'detector_axis',
        'recovery',
        'in_detector',
    ]
    if 'parent_pulse_length' in hit_table.columns:
        keep_columns.append('parent_pulse_length')
    hit_table = hit_table[keep_columns].reset_index(drop=True)

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


def _build_diagnostics_batch(args: tuple) -> list[dict]:
    """Process one batch of sequence records for ``build_surface_concept_recovery_diagnostics``.

    Defined at module level (not nested) so the ProcessPoolExecutor spawn
    workers can import and call it. ``args`` is a tuple to keep the worker
    signature single-arg, which lets us drive it with ``parallel_map``.
    """
    records, start_index, detector_limit_cm = args
    rows: list[dict] = []
    for offset, record in enumerate(records):
        sequence_index = start_index + offset
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
                        'parent_pulse_length': length,
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

            # Diagnostics mode: enumerate every per-axis pair (no detector
            # drop -- radius=inf) and keep the axes separate so the
            # acceptance flagging below can mark each hit in_detector
            # True/False against the caller's detector_limit_cm.
            partial_hits = _recover_surface_concept_partial_hits(
                chunk_channels,
                chunk_times,
                detector_radius_cm=float('inf'),
                combine_axes=False,
            )
            if not partial_hits:
                rows.append(
                    {
                        'sequence_index': sequence_index,
                        'chunk_index': chunk_index,
                        'parent_pulse_length': length,
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
                axis = str(partial_hit['detector_axis'])
                if axis == 'x':
                    in_detector = abs(det_x) <= detector_limit_cm
                elif axis == 'y':
                    in_detector = abs(det_y) <= detector_limit_cm
                else:
                    in_detector = abs(det_x) <= detector_limit_cm and abs(det_y) <= detector_limit_cm
                # A combined hit (detector_axis == 'xy') fired all four
                # delay-line ends -> 4 DLTS; a single-axis partial -> 2.
                # The diagnostics path runs with combine_axes=False so axis
                # is always 'x'/'y' here, but label correctly in case the
                # combine mode is ever used for diagnostics.
                dlts_value = 4 if axis == 'xy' else 2
                rows.append(
                    {
                        'sequence_index': sequence_index,
                        'chunk_index': chunk_index,
                        'parent_pulse_length': length,
                        'start_counter': start_counter,
                        'high_voltage (V)': high_voltage,
                        'pulse': pulse,
                        'tof (ns)': float(partial_hit['tof (ns)']),
                        'x_det (cm)': det_x,
                        'y_det (cm)': det_y,
                        'radius_cm': float(np.hypot(det_x, det_y)),
                        'dlts': dlts_value,
                        'detector_axis': axis,
                        'accepted': in_detector,
                        'status': f'{dlts_value} DLTS in detector' if in_detector else f'{dlts_value} DLTS outside detector',
                    }
                )
    return rows


def build_surface_concept_recovery_diagnostics(
    sequence_records: list[dict],
    *,
    detector_limit_cm: float = 4.0,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Build a per-candidate recovery table for advanced Surface Concept diagnostics.

    The per-record work is pure Python (dict construction, classification),
    so we farm batches of records to a ProcessPool via :func:`parallel_map`
    (``gil_releasing=False``). Below the auto-serial threshold, this is the
    same in-process loop as before; for >=200 records the wall-clock drops
    proportionally to ``PYCCAPT_PARALLEL_WORKERS``.
    """
    from pyccapt.calibration.core.parallel import ParallelConfig, parallel_map

    n_records = len(sequence_records)
    if n_records == 0:
        return pd.DataFrame()

    # Measured break-even: ProcessPool only beats serial above ~500 k records
    # for this data shape (list of dicts of lists). At smaller sizes the
    # pickle/IPC cost of shipping batches between workers exceeds the inner
    # compute. Below the threshold we run the worker in-process. The
    # threshold can be lowered via PYCCAPT_PARALLEL_WORKERS env tweaks when
    # users have a faster IPC path (e.g. fork on Linux).
    PROCESS_ENGAGE_THRESHOLD = 500_000
    if n_records < PROCESS_ENGAGE_THRESHOLD:
        return pd.DataFrame(_build_diagnostics_batch((sequence_records, 0, detector_limit_cm)))

    batch_size = max(500, n_records // 16)
    batches: list[tuple] = []
    for start in range(0, n_records, batch_size):
        batches.append((sequence_records[start : start + batch_size], start, detector_limit_cm))

    progress = None
    if show_progress:
        progress = tqdm(total=n_records, desc='Recovering Surface Concept hits', unit='sequence')

    batch_config = ParallelConfig(min_items=2)
    batch_results = parallel_map(
        _build_diagnostics_batch,
        batches,
        config=batch_config,
        gil_releasing=False,
    )

    rows: list[dict] = []
    for batch_rows in batch_results:
        rows.extend(batch_rows)
        if progress is not None:
            progress.update(batch_size)
    if progress is not None:
        progress.close()
    return pd.DataFrame(rows)


def analyze_surface_concept_tdc_frame(
    df_tdc: pd.DataFrame,
    *,
    detector_limit_cm: float = 4.0,
    t0: float = 0.0,
    flight_path_length_mm: float = 110.0,
    pulse_mode: str = 'voltage',
    show_progress: bool = False,
    sequence_records: list | None = None,
) -> dict:
    """Recover and analyze Surface Concept hits from an already-loaded raw tdc frame.

    Pass ``sequence_records`` to skip the (expensive) call to
    ``find_consecutive_sequences`` when the caller already holds pre-computed
    records (e.g. from a preceding combinatorial analysis).
    """
    required = {'start_counter', 'channel', 'time_data', 'high_voltage (V)'}
    missing = required.difference(df_tdc.columns)
    if missing:
        raise ValueError(f"Surface Concept tdc frame is missing required columns: {sorted(missing)}")

    if sequence_records is None:
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
            positions = (
                subset['x_det (cm)'].to_numpy(dtype=float) if axis_name == 'x' else subset['y_det (cm)'].to_numpy(dtype=float)
            )
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
        summary = pd.DataFrame(
            {'label': ordered_labels, 'dlts': np.zeros(len(ordered_labels)), 'count': np.zeros(len(ordered_labels))}
        )

    def _count_for(label: str, dlts: int) -> int:
        matches = summary[(summary['label'] == label) & (summary['dlts'] == dlts)]
        if matches.empty:
            return 0
        return int(matches['count'].sum())

    total_in_detector = int(
        np.count_nonzero(
            (hit_table['dlts'].isin([2, 4]).to_numpy())
            & (
                hit_table['in_detector'].to_numpy()
                if 'in_detector' in hit_table.columns
                else np.ones(len(hit_table), dtype=bool)
            )
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


# ---------------------------------------------------------------------------
# Per-peak diagnostics: chunk-length distribution + detector-position plots
# ---------------------------------------------------------------------------


def _resolve_peak_signal_column(hit_table: pd.DataFrame, signal_kind: str) -> str:
    signal_kind = _normalize_signal_kind(signal_kind)
    if signal_kind == 'mc':
        return 'mc (Da)'
    return 'tof (ns)'


def filter_peak_hits(
    hit_table: pd.DataFrame,
    window: dict,
    *,
    signal_kind: str = 'tof',
    only_in_detector: bool = True,
) -> pd.DataFrame:
    """Return the rows of ``hit_table`` whose chosen signal lies in ``window``.

    ``window`` is a dict with ``min``/``max`` keys (the format produced by
    :func:`pyccapt.calibration.data_tools._raw_workflow_common.normalize_signal_windows`).
    Setting ``only_in_detector=True`` (default) drops hits that failed the
    per-axis detector-area check from
    :func:`build_surface_concept_recovery_diagnostics`.
    """
    if hit_table is None or hit_table.empty:
        return hit_table.iloc[0:0] if hit_table is not None else pd.DataFrame()
    column = _resolve_peak_signal_column(hit_table, signal_kind)
    if column not in hit_table.columns:
        return hit_table.iloc[0:0]

    frame = hit_table
    if only_in_detector and 'in_detector' in frame.columns:
        frame = frame[frame['in_detector']]
    values = pd.to_numeric(frame[column], errors='coerce').to_numpy()
    lower = float(window['min'])
    upper = float(window['max'])
    mask = (values >= lower) & (values <= upper)
    return frame.loc[mask].copy()


def plot_peak_chunk_length_distribution(
    hit_table: pd.DataFrame,
    windows: Sequence[dict],
    *,
    signal_kind: str = 'tof',
    only_in_detector: bool = True,
    max_length: int = 20,
) -> plt.Figure | None:
    """For every user peak window, render two side-by-side bars per parent
    chunk length: how many *2-DLTS partial* hits and *4-DLTS full* hits in
    that peak came from a parent pulse of that length.

    ``parent_pulse_length`` must be present on ``hit_table`` (it is added by
    :func:`build_surface_concept_recovery_diagnostics`).  Returns ``None`` if
    the column is missing or no peak windows are supplied.
    """
    if hit_table is None or hit_table.empty or 'parent_pulse_length' not in hit_table.columns:
        return None
    if not windows:
        return None

    n_peaks = len(windows)
    # Cap the figure height well below matplotlib's 2^16-pixel hard limit so
    # IPython's ``bbox_inches='tight'`` PNG render can't blow the dimensions
    # into the multi-million-pixel range that triggered "Image size too large"
    # on user input.  20 in × 300 dpi = 6 000 px — comfortably below 65 535.
    height_inches = min(max(2.4 * n_peaks, 3.0), 20.0)
    fig, axes = plt.subplots(
        n_peaks,
        1,
        figsize=(8.0, height_inches),
        squeeze=False,
    )
    bins = np.arange(0.5, max_length + 1.5)
    centers = np.arange(1, max_length + 1)

    for index, window in enumerate(windows):
        ax = axes[index][0]
        peak = filter_peak_hits(hit_table, window, signal_kind=signal_kind, only_in_detector=only_in_detector)
        if peak.empty:
            ax.text(0.5, 0.5, "no hits in this peak window", ha='center', va='center', transform=ax.transAxes, color='gray')
            ax.set_xlim(0.5, max_length + 0.5)
            ax.set_xticks(centers)
            ax.set_title(f"{window.get('label', f'Peak {index + 1}')}")
            ax.set_xlabel("Parent pulse length (DLTS per pulse)")
            ax.set_ylabel("Hit count")
            continue

        partial = peak[peak['dlts'] == 2]['parent_pulse_length'].to_numpy()
        full = peak[peak['dlts'] == 4]['parent_pulse_length'].to_numpy()
        partial_hist = np.histogram(partial, bins=bins)[0]
        full_hist = np.histogram(full, bins=bins)[0]

        w = 0.4
        ax.bar(centers - 0.5 * w, partial_hist, width=w, color=DLTS_COLORS.get(2, '#f59e0b'), label='2 DLTS')
        ax.bar(centers + 0.5 * w, full_hist, width=w, color=DLTS_COLORS.get(4, '#1f77b4'), label='4 DLTS')
        ax.set_yscale('log')
        ax.set_xlim(0.5, max_length + 0.5)
        ax.set_xticks(centers)
        ax.set_xlabel("Parent pulse length (DLTS per pulse)")
        ax.set_ylabel("Hit count (log)")
        ax.set_title(
            f"{window.get('label', f'Peak {index + 1}')} — partial = {int(partial_hist.sum()):,}; "
            f"full = {int(full_hist.sum()):,}"
        )
        ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Combinatorial per-pulse hit recovery (greedy / exhaustive).
#
# The pre-existing recovery walks chunks-of-4 and applies a fixed pairing
# heuristic (``first ch=0`` with ``first ch=1``, etc.). It does not test
# whether each pair lands inside the detector or in the user's peak window
# before committing to the pairing — those checks happen afterwards.
#
# The combinatorial recovery below enumerates every reasonable
# ``(ch_a, ch_b)`` partial-hit pair and every ``(ch=0, ch=1, ch=2, ch=3)``
# complete-hit quadruple whose x-pair-sum and y-pair-sum coincide within a
# user-set tolerance (default ±200 TDC bins, ≈1.4 ns at 6.86 ps/bin), then
# selects the maximum set of valid hits whose timestamps don't overlap.
# Two selection modes are offered:
#
#   - ``greedy`` (default, fast): O(N²) — rank candidates by validity then
#     by signal distance to the nearest peak centre, pick the highest-ranked
#     one whose timestamps are still free, repeat. Matches the legacy
#     "first-occurrence" approach when no peak windows are supplied.
#   - ``exhaustive`` (slow, opt-in): branch-and-bound search of the
#     index-disjoint subsets of valid candidates; returns the largest one.
#     Falls back to greedy when the candidate count exceeds
#     ``exhaustive_max_candidates`` (default 80) so the worst case stays
#     bounded.
# ---------------------------------------------------------------------------


_DEFAULT_PAIR_SUM_TOLERANCE_BINS = 200.0
_DEFAULT_EXHAUSTIVE_MAX_CANDIDATES = 80


def _signal_in_any_window(signal_value: float, peak_windows: Sequence[dict] | None) -> bool:
    """Return ``True`` if ``signal_value`` lies inside any user-defined peak
    window. With no windows supplied the predicate is vacuously true (the
    user hasn't gated on signal yet)."""
    if not peak_windows:
        return True
    for window in peak_windows:
        if float(window['min']) <= signal_value <= float(window['max']):
            return True
    return False


def _hit_in_detector_axis_aware(det_x: float, det_y: float, axis: str, limit_cm: float) -> bool:
    """Detector-area validity gate.

    Partial 2-DLTS hits only have one reconstructed coordinate, so only that
    axis is checked. Complete 4-DLTS hits need both axes inside the detector.
    """
    if axis == 'x':
        return abs(det_x) <= limit_cm
    if axis == 'y':
        return abs(det_y) <= limit_cm
    return abs(det_x) <= limit_cm and abs(det_y) <= limit_cm


def _signal_distance_to_nearest_peak(
    signal_value: float,
    peak_windows: Sequence[dict] | None,
) -> float:
    """Distance from ``signal_value`` to the centre of its closest peak window.

    Used as a tie-breaker in the greedy selector — when several valid
    candidates compete for the same timestamp, we prefer the one whose
    signal is best-centered in some peak. Returns ``+inf`` when no windows
    are supplied or when ``signal_value`` is outside every window.
    """
    if not peak_windows:
        return float('inf')
    best = float('inf')
    for window in peak_windows:
        lo = float(window['min'])
        hi = float(window['max'])
        if lo <= signal_value <= hi:
            distance = abs(signal_value - 0.5 * (lo + hi))
            if distance < best:
                best = distance
    return best


def _generate_partial_candidates_for_axis(
    channels: np.ndarray,
    times: np.ndarray,
    axis: str,
    low_channel: int,
    high_channel: int,
    *,
    xy_factor: float,
    xy_bin_shift: float,
    tof_factor_2d: float,
) -> list[dict]:
    """Enumerate every ``(low_ch, high_ch)`` index pair as a candidate
    partial hit on the given axis."""
    candidates: list[dict] = []
    low_indices = np.where(channels == low_channel)[0].tolist()
    high_indices = np.where(channels == high_channel)[0].tolist()
    for i_low in low_indices:
        for i_high in high_indices:
            t_low = float(times[i_low])
            t_high = float(times[i_high])
            position = _position_from_pair(t_low, t_high, xy_factor, xy_bin_shift)
            tof = (t_low + t_high) * tof_factor_2d
            candidates.append(
                {
                    'dlts': 2,
                    'detector_axis': axis,
                    'used_indices': frozenset({int(i_low), int(i_high)}),
                    'x_det (cm)': position if axis == 'x' else 0.0,
                    'y_det (cm)': position if axis == 'y' else 0.0,
                    'tof (ns)': float(tof),
                    'pair_sum_bins_x': float(t_low + t_high) if axis == 'x' else None,
                    'pair_sum_bins_y': float(t_low + t_high) if axis == 'y' else None,
                }
            )
    return candidates


def _position_from_pair(first_time: float, second_time: float, xy_factor: float, xy_bin_shift: float) -> float:
    """Same algebra as :func:`_surface_concept_position_from_pair` but with
    explicit constants so the per-rig config can override them."""
    difference = second_time - first_time
    shifted = -0.5 * difference + xy_bin_shift
    return ((shifted - xy_bin_shift) * xy_factor) * 0.1


def _generate_complete_candidates(
    channels: np.ndarray,
    times: np.ndarray,
    *,
    xy_factor: float,
    xy_bin_shift: float,
    tof_factor_4d: float,
    pair_sum_tolerance_bins: float,
) -> list[dict]:
    """Enumerate complete-event (ch=0, ch=1, ch=2, ch=3) candidates whose
    x-pair-sum and y-pair-sum coincide within the tolerance.

    Two timestamps from the same physical ion satisfy
    ``(t0 + t1)/2 ≈ (t2 + t3)/2 = ion_arrival_time``, i.e.
    ``|sum_x - sum_y| ≤ tolerance``. This filter is what lets the
    combinatorial recovery distinguish a real complete event from a chance
    coincidence of two different ions both firing partial pairs.
    """
    candidates: list[dict] = []
    idx_0 = np.where(channels == 0)[0].tolist()
    idx_1 = np.where(channels == 1)[0].tolist()
    idx_2 = np.where(channels == 2)[0].tolist()
    idx_3 = np.where(channels == 3)[0].tolist()
    if not (idx_0 and idx_1 and idx_2 and idx_3):
        return candidates

    for i0 in idx_0:
        t0 = float(times[i0])
        for i1 in idx_1:
            t1 = float(times[i1])
            sum_x = t0 + t1
            for i2 in idx_2:
                t2 = float(times[i2])
                for i3 in idx_3:
                    t3 = float(times[i3])
                    sum_y = t2 + t3
                    if abs(sum_x - sum_y) > pair_sum_tolerance_bins:
                        continue
                    det_x = _position_from_pair(t0, t1, xy_factor, xy_bin_shift)
                    det_y = _position_from_pair(t2, t3, xy_factor, xy_bin_shift)
                    tof = (t0 + t1 + t2 + t3) * tof_factor_4d
                    candidates.append(
                        {
                            'dlts': 4,
                            'detector_axis': 'xy',
                            'used_indices': frozenset({int(i0), int(i1), int(i2), int(i3)}),
                            'x_det (cm)': det_x,
                            'y_det (cm)': det_y,
                            'tof (ns)': float(tof),
                            'pair_sum_bins_x': sum_x,
                            'pair_sum_bins_y': sum_y,
                        }
                    )
    return candidates


def _score_candidate_validity(
    candidate: dict,
    *,
    peak_windows: Sequence[dict] | None,
    detector_limit_cm: float,
    signal_kind: str,
    max_tof_ns: float = 5000.0,
) -> None:
    """Mutate ``candidate`` in place with the validity booleans the selector
    needs.

    A candidate is *valid for emission* when it is geometrically real:

    - the reconstructed coordinate is inside the detector face, AND
    - its TOF is in ``[0, max_tof_ns]`` and finite.

    The peak-window membership (``in_peak``) is computed too, but is
    informational — it is **not** part of ``candidate['valid']``. That
    means the recovery emits every geometrically valid hit, including
    noise events that fall between the user's peak windows. Per-peak
    yield tables / diagnostic plots downstream then filter the emitted
    hit table by peak window themselves.

    The previous policy ANDed ``in_peak`` into validity, which deleted
    the noise from the recovered hit table — so the "Full spectrum" view
    showed only the peak regions, defeating its purpose.
    """
    candidate['in_detector'] = _hit_in_detector_axis_aware(
        candidate['x_det (cm)'],
        candidate['y_det (cm)'],
        candidate['detector_axis'],
        detector_limit_cm,
    )
    tof_value = candidate.get('tof (ns)')
    if tof_value is None or not np.isfinite(tof_value):
        candidate['in_tof_range'] = False
    else:
        candidate['in_tof_range'] = bool(0.0 <= float(tof_value) <= max_tof_ns)

    column = 'mc (Da)' if signal_kind == 'mc' else 'tof (ns)'
    signal_value = candidate.get(column)
    if signal_value is None or not np.isfinite(signal_value):
        candidate['in_peak'] = False
        candidate['signal_distance'] = float('inf')
    else:
        candidate['in_peak'] = _signal_in_any_window(float(signal_value), peak_windows)
        candidate['signal_distance'] = _signal_distance_to_nearest_peak(float(signal_value), peak_windows)

    # Validity for SELECTION = geometric only. Peak windows are applied
    # later, by the peak-yield helpers, against the full hit table.
    candidate['valid'] = bool(candidate['in_detector'] and candidate['in_tof_range'])


def _select_max_disjoint_greedy(candidates: list[dict]) -> list[dict]:
    """Greedy max-disjoint set on a homogeneous candidate list. Sorts by
    ``signal_distance`` ascending (tighter peak match wins ties) and picks
    the highest-ranked candidate whose timestamps are still free."""
    ranked = sorted(
        candidates,
        key=lambda c: float(c.get('signal_distance', float('inf'))),
    )
    used: set[int] = set()
    emitted: list[dict] = []
    for candidate in ranked:
        if candidate['used_indices'].isdisjoint(used):
            emitted.append(candidate)
            used.update(candidate['used_indices'])
    return emitted


def _select_max_disjoint_exhaustive(
    candidates: list[dict],
    *,
    max_candidates: int = _DEFAULT_EXHAUSTIVE_MAX_CANDIDATES,
) -> list[dict]:
    """Branch-and-bound max-index-disjoint subset on a homogeneous candidate
    list. Falls back to greedy when the candidate count exceeds
    ``max_candidates`` so the worst case stays bounded."""
    if len(candidates) > max_candidates:
        return _select_max_disjoint_greedy(candidates)

    ordered = sorted(
        candidates,
        key=lambda c: float(c.get('signal_distance', float('inf'))),
    )

    best_selection: list[int] = []

    def _solve(idx: int, used: frozenset[int], selection: list[int]) -> None:
        nonlocal best_selection
        # Branch-and-bound: even if we took every remaining candidate, can we
        # beat the current best? If not, prune.
        if len(selection) + (len(ordered) - idx) <= len(best_selection):
            return
        if idx == len(ordered):
            if len(selection) > len(best_selection):
                best_selection = list(selection)
            return
        candidate = ordered[idx]
        if candidate['used_indices'].isdisjoint(used):
            selection.append(idx)
            _solve(idx + 1, used | candidate['used_indices'], selection)
            selection.pop()
        _solve(idx + 1, used, selection)

    _solve(0, frozenset(), [])
    return [ordered[i] for i in best_selection]


def _select_hits_two_stage(
    valid_candidates: list[dict],
    *,
    mode: str,
    exhaustive_max_candidates: int = _DEFAULT_EXHAUSTIVE_MAX_CANDIDATES,
) -> list[dict]:
    """Two-stage selection: COMPLETES first, then PARTIALS on remaining indices.

    The user-facing rule is "always when all channels available we have to
    first check full hit possibility then partial hit if full hit is not
    valid." A clean ``[0, 1, 2, 3]`` chunk has

    - 1 valid complete (4-DLTS) candidate, and
    - 1 valid x-partial + 1 valid y-partial that overlap exactly the complete
      on indices.

    Optimising raw hit *count* picks the 2 partials (count = 2) over the
    single complete (count = 1) — but physically that's one ion firing all
    four channels, so the right answer is **1 complete**. Two-stage
    selection enforces that:

    - **Phase 1:** find the maximum index-disjoint subset of *valid complete*
      candidates (greedy or exhaustive as the caller requests). Commit those
      indices.
    - **Phase 2:** filter valid partial candidates to those whose indices
      don't collide with the chosen completes; pick the max-disjoint subset
      of *those*.

    This guarantees every valid complete is locked in before any partial
    decomposition is considered, which is what the user asked for.
    """
    completes = [c for c in valid_candidates if int(c.get('dlts', 0)) == 4]
    partials = [c for c in valid_candidates if int(c.get('dlts', 0)) != 4]

    selector = _select_max_disjoint_exhaustive if str(mode).lower() == 'exhaustive' else _select_max_disjoint_greedy

    if str(mode).lower() == 'exhaustive':
        chosen_completes = _select_max_disjoint_exhaustive(
            completes,
            max_candidates=exhaustive_max_candidates,
        )
    else:
        chosen_completes = _select_max_disjoint_greedy(completes)

    used = frozenset()
    if chosen_completes:
        used = frozenset().union(*(c['used_indices'] for c in chosen_completes))

    available_partials = [p for p in partials if p['used_indices'].isdisjoint(used)]
    if str(mode).lower() == 'exhaustive':
        chosen_partials = _select_max_disjoint_exhaustive(
            available_partials,
            max_candidates=exhaustive_max_candidates,
        )
    else:
        chosen_partials = _select_max_disjoint_greedy(available_partials)

    return list(chosen_completes) + list(chosen_partials)


# Backwards-compatible aliases — the older single-stage selectors are kept
# under their previous names so any direct callers (and the existing tests)
# continue to work, but they now route through the two-stage logic so the
# completes-first contract is uniform across modes.


def _select_hits_greedy(valid_candidates: list[dict]) -> list[dict]:
    return _select_hits_two_stage(valid_candidates, mode='greedy')


def _select_hits_exhaustive(
    valid_candidates: list[dict],
    *,
    max_candidates: int = _DEFAULT_EXHAUSTIVE_MAX_CANDIDATES,
) -> list[dict]:
    return _select_hits_two_stage(
        valid_candidates,
        mode='exhaustive',
        exhaustive_max_candidates=max_candidates,
    )


def _compute_mc_for_candidates(
    candidates: list[dict],
    *,
    high_voltage: float,
    pulse_v: float,
    flight_path_length_mm: float,
    pulse_mode: str,
    t0: float,
) -> None:
    """Fill the ``mc (Da)`` field on every candidate using the same uncalibrated
    ``tof2mc`` formula the legacy notebook used (``t0=0``, ``V_pulse=zeros``,
    ``fpl=110 mm`` by default; overridable per call)."""
    if not candidates:
        return
    tof = np.array([float(c['tof (ns)']) for c in candidates])
    x = np.array([float(c['x_det (cm)']) for c in candidates])
    y = np.array([float(c['y_det (cm)']) for c in candidates])
    n = len(candidates)
    voltage = np.full(n, float(high_voltage))
    pulse_arr = np.full(n, float(pulse_v)) if pulse_mode == 'voltage' else np.zeros(n)
    mc_values = mc_tools.tof2mc(
        t=tof,
        t0=t0,
        V=voltage,
        xDet=x,
        yDet=y,
        flightPathLength=flight_path_length_mm,
        V_pulse=pulse_arr,
        mode=pulse_mode,
    )
    for candidate, mc_value in zip(candidates, mc_values):
        candidate['mc (Da)'] = float(mc_value)


def extract_valid_hits_combinatorial(
    record: dict,
    *,
    peak_windows: Sequence[dict] | None = None,
    signal_kind: str = 'tof',
    detector_limit_cm: float = 4.0,
    max_tof_ns: float = 5000.0,
    mode: str = 'greedy',
    pair_sum_tolerance_bins: float = _DEFAULT_PAIR_SUM_TOLERANCE_BINS,
    flight_path_length_mm: float = 110.0,
    pulse_mode: str = 'voltage',
    t0: float = 0.0,
    xy_factor: float = XY_FACTOR,
    xy_bin_shift: float = XY_BIN_SHIFT,
    tof_factor_4d: float = TOF_FACTOR_NS,
    tof_factor_2d: float = TOF_FACTOR_NS_1D,
    exhaustive_max_candidates: int = _DEFAULT_EXHAUSTIVE_MAX_CANDIDATES,
) -> tuple[list[dict], list[dict]]:
    """Per-pulse combinatorial hit recovery for one ``find_consecutive_sequences``
    record.

    Returns ``(emitted_hits, all_candidates)``. Each emitted hit is annotated
    with ``parent_pulse_length``, ``high_voltage (V)``, ``pulse``,
    ``start_counter``, ``in_detector``, ``in_peak`` and ``valid``. The full
    candidate list (validity-scored, including ones that failed the validity
    gate or lost to a winning conflict) is returned alongside so diagnostics
    can show how many candidates were considered per pulse.
    """
    channels = np.asarray(record.get('channels', []), dtype=np.int64)
    times = np.asarray(record.get('time_data', []), dtype=np.int64)
    if channels.size == 0:
        return [], []

    high_voltage = float(record.get('high_voltage', 0.0))
    pulse_v = float(record.get('pulse', 0.0))
    start_counter_array = record.get('start_counter', [])
    start_counter = int(start_counter_array[0]) if len(start_counter_array) else 0
    pulse_length = int(channels.size)

    complete_candidates = _generate_complete_candidates(
        channels,
        times,
        xy_factor=xy_factor,
        xy_bin_shift=xy_bin_shift,
        tof_factor_4d=tof_factor_4d,
        pair_sum_tolerance_bins=pair_sum_tolerance_bins,
    )
    partial_x = _generate_partial_candidates_for_axis(
        channels,
        times,
        'x',
        0,
        1,
        xy_factor=xy_factor,
        xy_bin_shift=xy_bin_shift,
        tof_factor_2d=tof_factor_2d,
    )
    partial_y = _generate_partial_candidates_for_axis(
        channels,
        times,
        'y',
        2,
        3,
        xy_factor=xy_factor,
        xy_bin_shift=xy_bin_shift,
        tof_factor_2d=tof_factor_2d,
    )
    all_candidates = complete_candidates + partial_x + partial_y

    # Always compute mc on every candidate so the validity gate can use either
    # ``tof`` or ``mc`` interchangeably and downstream consumers (e.g.
    # ``surface_concept_hits_to_processed_dataframe``) can build the
    # processed dataframe without a missing-column error.
    _compute_mc_for_candidates(
        all_candidates,
        high_voltage=high_voltage,
        pulse_v=pulse_v,
        flight_path_length_mm=flight_path_length_mm,
        pulse_mode=pulse_mode,
        t0=t0,
    )

    for candidate in all_candidates:
        _score_candidate_validity(
            candidate,
            peak_windows=peak_windows,
            detector_limit_cm=detector_limit_cm,
            signal_kind=signal_kind,
            max_tof_ns=max_tof_ns,
        )

    valid_candidates = [c for c in all_candidates if c['valid']]
    # Two-stage selection: lock valid completes first (in their max-disjoint
    # set), THEN fill valid partials on the remaining indices. This enforces
    # the user-stated rule "always when all channels available we have to
    # first check full hit possibility then partial hit if full hit is not
    # valid" — without it, the exhaustive mode prefers (count = 2 partials)
    # over (count = 1 complete) on a clean [0, 1, 2, 3] chunk.
    emitted = _select_hits_two_stage(
        valid_candidates,
        mode=mode.lower(),
        exhaustive_max_candidates=exhaustive_max_candidates,
    )

    for hit in emitted:
        hit['parent_pulse_length'] = pulse_length
        hit['high_voltage (V)'] = high_voltage
        hit['pulse'] = pulse_v
        hit['start_counter'] = start_counter

    return emitted, all_candidates


def _run_combinatorial_batch(args: tuple) -> tuple[list[dict], dict]:
    """Process one batch of sequence records for the combinatorial recovery.

    Must be at module level so ProcessPoolExecutor workers can import it.
    ``args = (records, kwargs)`` where ``kwargs`` contains all fixed
    parameters for :func:`extract_valid_hits_combinatorial`.

    Returns ``(rows, candidate_counts)``.
    """
    records, kwargs = args
    rows: list[dict] = []
    counts = {'total': 0, 'valid': 0, 'in_peak': 0, 'emitted': 0}
    for record in records:
        emitted, all_candidates = extract_valid_hits_combinatorial(record, **kwargs)
        counts['total'] += len(all_candidates)
        counts['valid'] += sum(1 for c in all_candidates if c['valid'])
        counts['in_peak'] += sum(1 for c in all_candidates if c.get('in_peak'))
        counts['emitted'] += len(emitted)
        rows.extend(
            {
                'start_counter': hit['start_counter'],
                'high_voltage (V)': hit['high_voltage (V)'],
                'pulse': hit['pulse'],
                'tof (ns)': hit['tof (ns)'],
                'mc (Da)': hit.get('mc (Da)', float('nan')),
                'x_det (cm)': hit['x_det (cm)'],
                'y_det (cm)': hit['y_det (cm)'],
                'dlts': int(hit['dlts']),
                'detector_axis': str(hit['detector_axis']),
                'recovery': f"{int(hit['dlts'])} DLTS",
                'in_detector': bool(hit['in_detector']),
                'in_peak': bool(hit['in_peak']),
                'parent_pulse_length': int(hit['parent_pulse_length']),
            }
            for hit in emitted
        )
    return rows, counts


def analyze_surface_concept_tdc_frame_combinatorial(
    df_tdc: pd.DataFrame,
    *,
    peak_windows: Sequence[dict] | None = None,
    signal_kind: str = 'tof',
    detector_limit_cm: float = 4.0,
    max_tof_ns: float = 5000.0,
    mode: str = 'greedy',
    pair_sum_tolerance_bins: float = _DEFAULT_PAIR_SUM_TOLERANCE_BINS,
    exhaustive_max_candidates: int = _DEFAULT_EXHAUSTIVE_MAX_CANDIDATES,
    t0: float = 0.0,
    flight_path_length_mm: float = 110.0,
    pulse_mode: str = 'voltage',
    show_progress: bool = False,
    xy_factor: float = XY_FACTOR,
    xy_bin_shift: float = XY_BIN_SHIFT,
    tof_factor_4d: float = TOF_FACTOR_NS,
    tof_factor_2d: float = TOF_FACTOR_NS_1D,
) -> dict:
    """Run the combinatorial per-pulse hit recovery on a Surface Concept tdc
    frame.

    This is the counterpart of :func:`analyze_surface_concept_tdc_frame` but
    drives pair selection by detector-area + peak-window validity (rather
    than by a fixed first-occurrence heuristic). Pass the user's peak
    windows in either TOF (ns) or mass/charge (Da) units; choose ``mode``
    ``'greedy'`` (default, fast O(N²)) or ``'exhaustive'`` (slow,
    branch-and-bound, falls back to greedy when a pulse generates more than
    ``exhaustive_max_candidates`` candidates).
    """
    required = {'start_counter', 'channel', 'time_data', 'high_voltage (V)'}
    missing = required.difference(df_tdc.columns)
    if missing:
        raise ValueError(f"Surface Concept tdc frame is missing required columns: {sorted(missing)}")

    pulse_column = _surface_concept_pulse_column(df_tdc, pulse_mode)

    # Use a generator so sequence records are produced one at a time and
    # immediately discarded after processing.  The old approach built a Python
    # list of ~16 M dicts which consumed ~20 GB of RAM and crashed the system.
    record_gen = raw_data_surface_concept.iter_consecutive_sequences(
        df_tdc['start_counter'].to_numpy(),
        df_tdc['channel'].to_numpy(),
        df_tdc['time_data'].to_numpy(),
        df_tdc['high_voltage (V)'].to_numpy(),
        df_tdc[pulse_column].to_numpy(),
        show_progress=show_progress,
    )

    # Accumulate hit columns as plain Python lists of scalars — much cheaper
    # than a list of dicts (no per-row dict/key overhead).
    col_sc: list = []
    col_hv: list = []
    col_pulse: list = []
    col_tof: list = []
    col_mc: list = []
    col_x: list = []
    col_y: list = []
    col_dlts: list = []
    col_axis: list = []
    col_recovery: list = []
    col_in_det: list = []
    col_in_peak: list = []
    col_plen: list = []

    candidate_counts = {'total': 0, 'valid': 0, 'in_peak': 0, 'emitted': 0}

    # Sequence / raw-summary accumulators (replaces summarize_surface_concept_sequences
    # and summarize_surface_concept_raw_sequences, computed in one pass).
    total_counts: Counter = Counter()
    dld2_counts: Counter = Counter()
    dld4_counts: Counter = Counter()
    invalid_counts: Counter = Counter()
    n_sequences = 0
    total_timestamps = 0
    channel_ts: Counter = Counter()
    valid_four = 0
    invalid_four = 0
    len3 = len2 = len1 = 0
    mh_four = mh_irreg = 0

    for record in record_gen:
        # --- sequence stats (incremental) ---
        ch_arr = np.asarray(record.get('channels', []), dtype=np.int64)
        td_arr = np.asarray(record.get('time_data', []), dtype=np.int64)
        length = int(ch_arr.size)
        n_sequences += 1
        total_timestamps += length
        channel_ts.update(int(c) for c in ch_arr.tolist())
        total_counts[length] += 1
        valid_events = list(record.get('valid_event', []))
        n_chunks = max(len(valid_events), math.ceil(length / 4))
        for ci in range(n_chunks):
            s, e = ci * 4, min(ci * 4 + 4, length)
            if s >= e:
                continue
            cc = ch_arr[s:e]
            ct = td_arr[s:e]
            is_valid = ci < len(valid_events) and bool(valid_events[ci]) and len(cc) == 4
            if is_valid:
                dld4_counts[length] += 1
            else:
                ph = _recover_surface_concept_partial_hits(cc, ct)
                if ph:
                    dld2_counts[length] += len(ph)
                else:
                    invalid_counts[length] += 1
        if length == 4:
            if valid_events == [True]:
                valid_four += 1
            else:
                invalid_four += 1
        elif length == 3:
            len3 += 1
        elif length == 2:
            len2 += 1
        elif length == 1:
            len1 += 1
        elif length > 4 and length % 4 == 0:
            mh_four += 1
        elif length > 4:
            mh_irreg += 1

        # --- combinatorial hit recovery ---
        emitted, all_candidates = extract_valid_hits_combinatorial(
            record,
            peak_windows=peak_windows,
            signal_kind=signal_kind,
            detector_limit_cm=detector_limit_cm,
            max_tof_ns=max_tof_ns,
            mode=mode,
            pair_sum_tolerance_bins=pair_sum_tolerance_bins,
            flight_path_length_mm=flight_path_length_mm,
            pulse_mode=pulse_mode,
            t0=t0,
            xy_factor=xy_factor,
            xy_bin_shift=xy_bin_shift,
            tof_factor_4d=tof_factor_4d,
            tof_factor_2d=tof_factor_2d,
            exhaustive_max_candidates=exhaustive_max_candidates,
        )
        candidate_counts['total'] += len(all_candidates)
        candidate_counts['valid'] += sum(1 for c in all_candidates if c['valid'])
        candidate_counts['in_peak'] += sum(1 for c in all_candidates if c.get('in_peak'))
        candidate_counts['emitted'] += len(emitted)
        for hit in emitted:
            col_sc.append(hit['start_counter'])
            col_hv.append(hit['high_voltage (V)'])
            col_pulse.append(hit['pulse'])
            col_tof.append(hit['tof (ns)'])
            col_mc.append(hit.get('mc (Da)', float('nan')))
            col_x.append(hit['x_det (cm)'])
            col_y.append(hit['y_det (cm)'])
            col_dlts.append(int(hit['dlts']))
            col_axis.append(str(hit['detector_axis']))
            col_recovery.append(f"{int(hit['dlts'])} DLTS")
            col_in_det.append(bool(hit['in_detector']))
            col_in_peak.append(bool(hit['in_peak']))
            col_plen.append(int(hit['parent_pulse_length']))

    gc.collect()

    if col_sc:
        hit_table = pd.DataFrame({
            'start_counter': col_sc,
            'high_voltage (V)': col_hv,
            'pulse': col_pulse,
            'tof (ns)': col_tof,
            'mc (Da)': col_mc,
            'x_det (cm)': col_x,
            'y_det (cm)': col_y,
            'dlts': col_dlts,
            'detector_axis': col_axis,
            'recovery': col_recovery,
            'in_detector': col_in_det,
            'in_peak': col_in_peak,
            'parent_pulse_length': col_plen,
        })
    else:
        hit_table = pd.DataFrame(columns=[
            'start_counter', 'high_voltage (V)', 'pulse', 'tof (ns)', 'mc (Da)',
            'x_det (cm)', 'y_det (cm)', 'dlts', 'detector_axis', 'recovery',
            'in_detector', 'in_peak', 'parent_pulse_length',
        ])

    # Free the column lists now that the DataFrame is built.
    del col_sc, col_hv, col_pulse, col_tof, col_mc, col_x, col_y
    del col_dlts, col_axis, col_recovery, col_in_det, col_in_peak, col_plen
    gc.collect()

    sequence_stats = {
        'total': dict(total_counts),
        'dld2': dict(dld2_counts),
        'dld4': dict(dld4_counts),
        'invalid': dict(invalid_counts),
    }
    raw_summary = {
        'total_sequences': n_sequences,
        'total_timestamps': total_timestamps,
        'channel_timestamp_totals': {ch: int(channel_ts[ch]) for ch in range(4)},
        'valid_four_channel_groups': valid_four,
        'invalid_four_channel_groups': invalid_four,
        'length_three_groups': len3,
        'length_two_groups': len2,
        'length_one_groups': len1,
        'multi_hit_groups_of_four': mh_four,
        'multi_hit_irregular': mh_irreg,
        'multi_hit_groups_of_four_timestamps': 0,
        'multi_hit_irregular_timestamps': 0,
    }

    # Derive recovery_stats from the hit_table (no second pass over raw data).
    if not hit_table.empty:
        dlts_col = hit_table['dlts'].to_numpy()
        in_det_col = hit_table['in_detector'].to_numpy()
        recovery_stats = {
            'recovered_hits': len(hit_table),
            'two_d_hits': int((dlts_col == 4).sum()),
            'one_d_hits': int((dlts_col == 2).sum()),
            'two_d_in_detector': int(((dlts_col == 4) & in_det_col).sum()),
            'one_d_in_detector': int(((dlts_col == 2) & in_det_col).sum()),
            'outside_detector_hits': int((~in_det_col).sum()),
            'unrecoverable_chunks': 0,
        }
    else:
        recovery_stats = {
            'recovered_hits': 0, 'two_d_hits': 0, 'one_d_hits': 0,
            'two_d_in_detector': 0, 'one_d_in_detector': 0,
            'outside_detector_hits': 0, 'unrecoverable_chunks': 0,
        }

    return {
        'hit_table': hit_table,
        'candidate_counts': candidate_counts,
        'sequence_stats': sequence_stats,
        'raw_summary': raw_summary,
        'recovery_diagnostics': pd.DataFrame(),
        'recovery_stats': recovery_stats,
        'mode': mode,
        'peak_windows': list(peak_windows) if peak_windows else [],
        'signal_kind': signal_kind,
        'pair_sum_tolerance_bins': pair_sum_tolerance_bins,
        'detector_limit_cm': detector_limit_cm,
    }


def plot_peak_detector_diagnostics(
    hit_table: pd.DataFrame,
    windows: Sequence[dict],
    *,
    signal_kind: str = 'tof',
    only_in_detector: bool = True,
    detector_limit_cm: float = 4.0,
    bin_size_cm: float = 0.1,
) -> plt.Figure | None:
    """Per-peak detector position diagnostics.

    For each peak window, render a 1-row × 3-column strip:

    - **2D detector hist** of *all* hits in the window (4-DLTS only contribute
      to the 2D map, since 2-DLTS hits have one coordinate set to zero and
      would smear into a line on the axis).
    - **1D x distribution** for hits whose recoverable axis is x — i.e. all
      4-DLTS hits and all 2-DLTS x-axis partials.
    - **1D y distribution** symmetric for the y axis.

    Returns ``None`` if there are no windows or the hit table is empty.
    """
    if hit_table is None or hit_table.empty:
        return None
    if not windows:
        return None

    n_peaks = len(windows)
    # Same safety cap as ``plot_peak_chunk_length_distribution``: matplotlib's
    # 2^16 px hard limit + IPython's ``bbox_inches='tight'`` PNG render path
    # was producing 46 M-px figures on long peak lists with
    # ``constrained_layout=True``. ``tight_layout`` (called below) is more
    # conservative; the cap further bounds the worst case.
    height_inches = min(max(3.0 * n_peaks, 3.4), 24.0)
    fig, axes = plt.subplots(
        n_peaks,
        3,
        figsize=(11.0, height_inches),
        squeeze=False,
    )
    edges = np.arange(-detector_limit_cm, detector_limit_cm + bin_size_cm, bin_size_cm)
    label_4 = '4 DLTS'
    label_2 = '2 DLTS'

    for index, window in enumerate(windows):
        peak_label = window.get('label', f'Peak {index + 1}')
        peak = filter_peak_hits(hit_table, window, signal_kind=signal_kind, only_in_detector=only_in_detector)
        ax_2d, ax_x, ax_y = axes[index]

        if peak.empty:
            for ax in (ax_2d, ax_x, ax_y):
                ax.text(0.5, 0.5, "no hits", ha='center', va='center', transform=ax.transAxes, color='gray')
            ax_2d.set_title(f"{peak_label}: 2D FDM")
            ax_x.set_title(f"{peak_label}: x distribution")
            ax_y.set_title(f"{peak_label}: y distribution")
            continue

        full_hits = peak[peak['dlts'] == 4]
        partial_x = peak[(peak['dlts'] == 2) & (peak['detector_axis'] == 'x')]
        partial_y = peak[(peak['dlts'] == 2) & (peak['detector_axis'] == 'y')]

        # 2D detector map: only full 4-DLTS hits have both coords meaningful.
        if not full_hits.empty:
            ax_2d.hist2d(
                full_hits['x_det (cm)'].to_numpy(),
                full_hits['y_det (cm)'].to_numpy(),
                bins=[edges, edges],
                cmap='viridis',
                norm=plt.matplotlib.colors.LogNorm(),
            )
        ax_2d.set_aspect('equal')
        ax_2d.set_xlim(-detector_limit_cm, detector_limit_cm)
        ax_2d.set_ylim(-detector_limit_cm, detector_limit_cm)
        ax_2d.set_xlabel('x_det (cm)')
        ax_2d.set_ylabel('y_det (cm)')
        ax_2d.set_title(f"{peak_label}: 2D FDM (4 DLTS)")

        # 1D x distribution: 4-DLTS hits + 2-DLTS x-axis partials.
        if not full_hits.empty:
            ax_x.hist(
                full_hits['x_det (cm)'].to_numpy(), bins=edges, color=DLTS_COLORS.get(4, '#1f77b4'), alpha=0.6, label=label_4
            )
        if not partial_x.empty:
            ax_x.hist(
                partial_x['x_det (cm)'].to_numpy(), bins=edges, color=DLTS_COLORS.get(2, '#f59e0b'), alpha=0.6, label=label_2
            )
        ax_x.set_yscale('log')
        ax_x.set_xlim(-detector_limit_cm, detector_limit_cm)
        ax_x.set_xlabel('x_det (cm)')
        ax_x.set_ylabel('Count (log)')
        ax_x.set_title(f"{peak_label}: x distribution")
        if not full_hits.empty or not partial_x.empty:
            ax_x.legend(fontsize=8, loc='upper right')

        # 1D y distribution: 4-DLTS hits + 2-DLTS y-axis partials.
        if not full_hits.empty:
            ax_y.hist(
                full_hits['y_det (cm)'].to_numpy(), bins=edges, color=DLTS_COLORS.get(4, '#1f77b4'), alpha=0.6, label=label_4
            )
        if not partial_y.empty:
            ax_y.hist(
                partial_y['y_det (cm)'].to_numpy(), bins=edges, color=DLTS_COLORS.get(2, '#f59e0b'), alpha=0.6, label=label_2
            )
        ax_y.set_yscale('log')
        ax_y.set_xlim(-detector_limit_cm, detector_limit_cm)
        ax_y.set_xlabel('y_det (cm)')
        ax_y.set_ylabel('Count (log)')
        ax_y.set_title(f"{peak_label}: y distribution")
        if not full_hits.empty or not partial_y.empty:
            ax_y.legend(fontsize=8, loc='upper right')

    fig.tight_layout()
    return fig
