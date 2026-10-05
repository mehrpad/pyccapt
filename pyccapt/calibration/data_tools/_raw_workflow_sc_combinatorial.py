"""Combinatorial Surface Concept recovery and diagnostics domain logic."""

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

from pyccapt.calibration.data_tools import raw_data_surface_concept
from pyccapt.calibration.data_tools._raw_workflow_common import (
    DLTS_COLORS, TOF_FACTOR_NS, TOF_FACTOR_NS_1D, XY_BIN_SHIFT, XY_FACTOR,
)
from pyccapt.calibration.data_tools._raw_workflow_surface_concept import (
    _recover_surface_concept_partial_hits, _surface_concept_pulse_column, filter_peak_hits,
)
from pyccapt.calibration.mc import mc_tools

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
