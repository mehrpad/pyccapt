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


def _recover_three_channel_hits(
    chunk_channels: np.ndarray,
    chunk_times: np.ndarray,
    used_global: set,
    x_leftover: list[dict],
    y_leftover: list[dict],
    detector_radius_cm: float,
    max_tof_ns: float | None,
) -> tuple[list[dict], set, set]:
    """Recover full xy hits from 3-channel pulses via the delay-line time-sum.

    A pulse that fired one complete delay-line axis (both ends) plus a
    single end of the OTHER axis is reconstructable. The two crossed delay
    lines share the same total propagation time, ``t0 + t1 = t2 + t3``
    (both start from the same MCP signal), so the missing end is

        ``t_missing = sum_complete_axis - t_present``

    and with it both det_x and det_y are computed -> a full 4-DLTS-equivalent
    xy hit. This is the offline analogue of the standard "3 of 4" delay-line
    reconstruction; the Surface Concept firmware quadrupel finder requires
    all four stops, so these hits are only recovered here.

    Each leftover complete-axis pair is matched with the most central
    coincident orphan single channel of the other axis (one orphan per
    pair). Hits are gated by non-negative recovered times, the detector
    radius, and the ToF window so a wrong orphan pairing is rejected.

    Returns ``(recovered_hits, consumed_x_idx, consumed_y_idx)`` -- the
    index sets mark which ``x_leftover`` / ``y_leftover`` entries were
    promoted, so the caller can drop them instead of emitting them again as
    1-D partials.
    """
    recovered: list[dict] = []
    consumed_x: set = set()
    consumed_y: set = set()
    n = chunk_channels.shape[0]

    orphan_x = [i for i in range(n) if i not in used_global and int(chunk_channels[i]) in (0, 1)]
    orphan_y = [i for i in range(n) if i not in used_global and int(chunk_channels[i]) in (2, 3)]
    if not orphan_x and not orphan_y:
        return recovered, consumed_x, consumed_y
    used_orphan: set = set()

    def _tof_ok(tof_val: float) -> bool:
        return max_tof_ns is None or (0.0 < tof_val <= max_tof_ns)

    # Complete Y axis + orphan X channel -> recover the missing X end.
    for yi, y_pair in enumerate(y_leftover):
        tof_full = float(y_pair['tof'])  # == sum_y * TOF_FACTOR_NS_1D (full tof)
        if not _tof_ok(tof_full):
            continue
        sum_y = tof_full / TOF_FACTOR_NS_1D
        best = None  # (abs_x, orphan_idx, x)
        for oi in orphan_x:
            if oi in used_orphan:
                continue
            channel = int(chunk_channels[oi])
            t_present = float(chunk_times[oi])
            if channel == 0:
                t0 = t_present
                t1 = sum_y - t0
            else:  # channel == 1
                t1 = t_present
                t0 = sum_y - t1
            if t0 < 0 or t1 < 0:
                continue
            x = _surface_concept_position_from_pair(t0, t1)
            if abs(x) > detector_radius_cm:
                continue
            if best is None or abs(x) < best[0]:
                best = (abs(x), oi, x)
        if best is not None:
            _, oi, x = best
            used_orphan.add(oi)
            consumed_y.add(yi)
            recovered.append({
                'x_det (cm)': float(x),
                'y_det (cm)': float(y_pair['position']),
                'tof (ns)': tof_full,
                'detector_axis': 'xy',
                'tof_axis_x_ns': tof_full,
                'tof_axis_y_ns': tof_full,
                'recovery_method': '3of4',
            })

    # Complete X axis + orphan Y channel -> recover the missing Y end.
    for xi, x_pair in enumerate(x_leftover):
        tof_full = float(x_pair['tof'])
        if not _tof_ok(tof_full):
            continue
        sum_x = tof_full / TOF_FACTOR_NS_1D
        best = None
        for oi in orphan_y:
            if oi in used_orphan:
                continue
            channel = int(chunk_channels[oi])
            t_present = float(chunk_times[oi])
            if channel == 2:
                t2 = t_present
                t3 = sum_x - t2
            else:  # channel == 3
                t3 = t_present
                t2 = sum_x - t3
            if t2 < 0 or t3 < 0:
                continue
            y = _surface_concept_position_from_pair(t2, t3)
            if abs(y) > detector_radius_cm:
                continue
            if best is None or abs(y) < best[0]:
                best = (abs(y), oi, y)
        if best is not None:
            _, oi, y = best
            used_orphan.add(oi)
            consumed_x.add(xi)
            recovered.append({
                'x_det (cm)': float(x_pair['position']),
                'y_det (cm)': float(y),
                'tof (ns)': tof_full,
                'detector_axis': 'xy',
                'tof_axis_x_ns': tof_full,
                'tof_axis_y_ns': tof_full,
                'recovery_method': '3of4',
            })

    return recovered, consumed_x, consumed_y


def _recover_surface_concept_partial_hits(
    chunk_channels: np.ndarray,
    chunk_times: np.ndarray,
    *,
    detector_radius_cm: float = 4.0,
    max_tof_ns: float | None = None,
    axis_consistency_ns: float | None = 5.0,
    combine_axes: bool = True,
    recover_three_channel: bool = True,
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

    ``recover_three_channel`` (default True, only active with
    ``combine_axes=True``) additionally promotes 3-channel pulses -- one
    complete axis plus one orphan end of the other axis -- to full xy hits
    via the delay-line time-sum constraint (see
    :func:`_recover_three_channel_hits`). These carry
    ``recovery_method='3of4'`` so the caller can label them distinctly.

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
    three_channel_hits: list[dict] = []
    if combine_axes:
        xy_hits, x_leftover, y_leftover = _combine_axes_into_xy_hits(
            per_axis_selected['x'],
            per_axis_selected['y'],
            axis_consistency_ns=axis_consistency_ns,
            max_tof_ns=max_tof_ns,
        )
        # 3-of-4 recovery: a complete axis + one orphan end of the other
        # axis reconstructs a full xy hit through the time-sum constraint.
        if recover_three_channel and (x_leftover or y_leftover):
            used_global: set = set()
            for axis_pairs in per_axis_selected.values():
                for pair in axis_pairs:
                    used_global.add(int(pair['first_global']))
                    used_global.add(int(pair['second_global']))
            three_channel_hits, consumed_x, consumed_y = _recover_three_channel_hits(
                chunk_channels,
                chunk_times,
                used_global,
                x_leftover,
                y_leftover,
                detector_radius_cm=detector_radius_cm,
                max_tof_ns=max_tof_ns,
            )
            if consumed_x:
                x_leftover = [p for i, p in enumerate(x_leftover) if i not in consumed_x]
            if consumed_y:
                y_leftover = [p for i, p in enumerate(y_leftover) if i not in consumed_y]
    else:
        # Diagnostics/reporting mode: keep the two delay-line axes
        # separate so every reconstructible per-axis pair is emitted as
        # its own 2-DLTS partial hit.
        xy_hits = []
        x_leftover = list(per_axis_selected['x'])
        y_leftover = list(per_axis_selected['y'])

    # --- Emit results ----------------------------------------------------
    # Full xy hits first (more physically informative), then the 3-of-4
    # time-sum recoveries, then leftover single-axis partials. Each dict
    # matches the schema the caller in partial_recovery.py expects:
    # x_det / y_det are NaN on the axis that was not recovered.
    recovered_hits.extend(xy_hits)
    recovered_hits.extend(three_channel_hits)
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




# Compatibility re-exports for the peak-analysis and persistence layer.
from pyccapt.calibration.data_tools._raw_workflow_sc_peaks import (
    summarize_surface_concept_peak_windows,
    plot_surface_concept_peak_breakdown,
    plot_surface_concept_peak_ratio_table,
    surface_concept_hits_to_processed_dataframe,
    reconstruct_surface_concept_dataset,
    _resolve_peak_signal_column,
    filter_peak_hits,
    plot_peak_chunk_length_distribution,
)




# Compatibility re-exports; implementation lives in the focused domain module.
from pyccapt.calibration.data_tools._raw_workflow_sc_combinatorial import (
    _signal_in_any_window,
    _hit_in_detector_axis_aware,
    _signal_distance_to_nearest_peak,
    _generate_partial_candidates_for_axis,
    _position_from_pair,
    _generate_complete_candidates,
    _score_candidate_validity,
    _select_max_disjoint_greedy,
    _select_max_disjoint_exhaustive,
    _select_hits_two_stage,
    _select_hits_greedy,
    _select_hits_exhaustive,
    _compute_mc_for_candidates,
    extract_valid_hits_combinatorial,
    _run_combinatorial_batch,
    analyze_surface_concept_tdc_frame_combinatorial,
    plot_peak_detector_diagnostics,
)
