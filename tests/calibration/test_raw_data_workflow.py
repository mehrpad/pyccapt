import matplotlib
import numpy as np
import pandas as pd

from pyccapt.calibration.data_tools import raw_data_workflow

matplotlib.use('Agg')


def test_roentdek_hit_flattening_and_window_summary():
    events = [
        {
            'event_number': 1,
            'channels': [
                {'channel': 1, 'num_values': 1, 'values': [1.0]},
                {'channel': 2, 'num_values': 1, 'values': [1.0]},
                {'channel': 3, 'num_values': 1, 'values': [1.0]},
                {'channel': 4, 'num_values': 1, 'values': [1.0]},
                {'channel': 5, 'num_values': 1, 'values': [1.0]},
                {'channel': 6, 'num_values': 1, 'values': [1.0]},
            ],
        },
        {
            'event_number': 2,
            'channels': [
                {'channel': 1, 'num_values': 1, 'values': [1.0]},
                {'channel': 2, 'num_values': 1, 'values': [1.0]},
            ],
        },
    ]
    events, counters = raw_data_workflow.classify_roentdek_events(events)
    assert counters['dld6'][6] == 1
    assert counters['dld2'][2] == 1

    numeric_table = np.zeros((2, 21))
    numeric_table[0, 6] = 1.5
    numeric_table[0, 7] = -0.5
    numeric_table[0, 8] = 150.0
    numeric_table[1, 6] = -1.0
    numeric_table[1, 7] = 0.75
    numeric_table[1, 8] = 420.0

    enriched = raw_data_workflow.attach_roentdek_measurements(events, numeric_table)
    hit_table = raw_data_workflow.roentdek_hits_to_dataframe(enriched)

    assert list(hit_table['dlts']) == [6, 2]
    assert np.allclose(hit_table['tof (ns)'].to_numpy(), [150.0, 420.0])

    summary = raw_data_workflow.summarize_signal_windows(
        hit_table,
        [{'label': 'early', 'min': 100.0, 'max': 200.0}],
        signal_kind='tof',
        only_in_detector=False,
    )
    early_6 = summary[(summary['label'] == 'early') & (summary['dlts'] == 6)]['count'].iloc[0]
    noise_2 = summary[(summary['label'] == 'Noise') & (summary['dlts'] == 2)]['count'].iloc[0]
    assert int(early_6) == 1
    assert int(noise_2) == 1

    raw_summary = raw_data_workflow.summarize_roentdek_raw_events(events)
    assert raw_summary['matched_pattern_events'] == 2
    assert raw_summary['multi_hit_events'] == 0
    assert raw_summary['channel_timestamp_totals'][1] == 2


def test_signal_windows_reject_overlap():
    try:
        raw_data_workflow.normalize_signal_windows(
            [
                {'label': 'a', 'min': 10.0, 'max': 20.0},
                {'label': 'b', 'min': 19.0, 'max': 30.0},
            ]
        )
    except ValueError as exc:
        assert 'overlap' in str(exc).lower()
    else:
        raise AssertionError('Expected overlapping windows to raise a ValueError')


def test_surface_concept_recovery_and_processed_dataframe():
    sequence_records = [
        {
            'channels': [0, 1, 2, 3],
            'time_data': [100, 110, 120, 130],
            'start_counter': [10],
            'valid_event': [True],
            'high_voltage': 5000.0,
            'pulse': 0.0,
        },
        {
            'channels': [0, 1],
            'time_data': [200, 210],
            'start_counter': [10],
            'valid_event': [False],
            'high_voltage': 5000.0,
            'pulse': 0.0,
        },
    ]

    sequence_stats = raw_data_workflow.summarize_surface_concept_sequences(sequence_records)
    assert sequence_stats['dld4'][4] == 1
    assert sequence_stats['dld2'][2] == 1

    hit_table, recovery = raw_data_workflow.extract_surface_concept_hits(sequence_records, detector_limit_cm=10.0)
    assert recovery['recovered_hits'] == 2
    assert list(hit_table['dlts']) == [4, 2]

    hit_table['mc (Da)'] = np.array([12.0, 24.0])
    processed = raw_data_workflow.surface_concept_hits_to_processed_dataframe(hit_table, pulse_mode='voltage')

    assert list(processed['mc (Da)']) == [12.0, 24.0]
    assert list(processed['multi']) == [2, 2]
    assert 't (ns)' in processed.columns

    raw_summary = raw_data_workflow.summarize_surface_concept_raw_sequences(sequence_records)
    assert raw_summary['valid_four_channel_groups'] == 1
    assert raw_summary['length_two_groups'] == 1


def test_surface_concept_recovery_emits_one_x_and_one_y_when_both_pairs_present():
    """An invalid 4-DLTS chunk that contains BOTH the (0,1) and (2,3) pairs
    must emit two partial hits — one for each delay-line axis. This lets us
    keep ions that fired both delay lines but missed the 4-channel canonical
    sort (e.g. [0, 2, 1, 3] in TDC order, multi-hit splits, etc.)."""
    sequence_records = [
        {
            'channels': [0, 1, 2, 3],
            'time_data': [100, 110, 120, 130],
            'start_counter': [10],
            # Force the chunk through the partial-recovery path so we can
            # inspect both axes.
            'valid_event': [False],
            'high_voltage': 5000.0,
            'pulse': 0.0,
        },
    ]

    hit_table, recovery = raw_data_workflow.extract_surface_concept_hits(sequence_records, detector_limit_cm=10.0)

    assert recovery['recovered_hits'] == 2
    assert recovery['one_d_hits'] == 2
    assert sorted(hit_table['detector_axis'].tolist()) == ['x', 'y']


def test_surface_concept_recovery_falls_back_to_y_when_no_x_pair():
    """When the (0, 1) pair is not present but (2, 3) is, recovery must
    still produce a y-axis hit on its own."""
    sequence_records = [
        {
            'channels': [0, 2, 3, 0],  # has ch0 + ch2 + ch3 but no ch1
            'time_data': [100, 120, 130, 105],
            'start_counter': [10],
            'valid_event': [False],
            'high_voltage': 5000.0,
            'pulse': 0.0,
        },
    ]

    hit_table, recovery = raw_data_workflow.extract_surface_concept_hits(sequence_records, detector_limit_cm=10.0)

    assert recovery['recovered_hits'] == 1
    assert recovery['one_d_hits'] == 1
    assert hit_table['detector_axis'].tolist() == ['y']


def test_surface_concept_recovery_emits_one_hit_per_pair_in_multi_hit_chunk():
    """A multi-hit chunk like [0, 0, 1, 1] has two physical x-pairs (one per
    ion). Recovery must emit BOTH so neither ion is silently dropped — the
    user-facing 2-DLTS yield should reflect actual reconstructible events."""
    sequence_records = [
        {
            'channels': [0, 0, 1, 1],
            'time_data': [100, 102, 110, 112],
            'start_counter': [10],
            'valid_event': [False],
            'high_voltage': 5000.0,
            'pulse': 0.0,
        },
    ]

    hit_table, recovery = raw_data_workflow.extract_surface_concept_hits(sequence_records, detector_limit_cm=10.0)

    assert recovery['recovered_hits'] == 2
    assert recovery['one_d_hits'] == 2
    assert hit_table['detector_axis'].tolist() == ['x', 'x']


def test_surface_concept_recovery_validates_axis_specific_detector_bounds():
    """A 2-DLTS x-axis partial whose reconstructed x is OUTSIDE the detector
    must be flagged ``in_detector=False``; the y coordinate is unset (0.0)
    and is NOT used as a rejection criterion. Similarly for a y-axis partial.
    """
    # Build two chunks: one that lands inside the detector, one that lands
    # well outside on the x axis. detector_limit_cm = 0.05 cm (tight bound)
    # makes the second chunk's reconstructed |det_x| exceed the bound.
    sequence_records = [
        {
            'channels': [0, 1],
            'time_data': [100, 110],  # |det_x| ≈ small → in_detector
            'start_counter': [10],
            'valid_event': [False],
            'high_voltage': 5000.0,
            'pulse': 0.0,
        },
        {
            'channels': [0, 1],
            'time_data': [100, 9000],  # huge time gap → |det_x| HUGE → out
            'start_counter': [11],
            'valid_event': [False],
            'high_voltage': 5000.0,
            'pulse': 0.0,
        },
    ]
    hit_table, _ = raw_data_workflow.extract_surface_concept_hits(sequence_records, detector_limit_cm=0.05)

    assert hit_table['detector_axis'].tolist() == ['x', 'x']
    assert hit_table['in_detector'].tolist() == [True, False]


def test_surface_concept_recovery_carries_parent_pulse_length_on_hit_table():
    """Each row of the hit table must remember the *parent pulse length* it
    came from, so per-peak diagnostics can answer "this 2-DLTS hit was
    recovered from a length-N pulse"."""
    sequence_records = [
        {
            'channels': [0, 1, 2, 3],
            'time_data': [100, 110, 120, 130],
            'start_counter': [10],
            'valid_event': [True],  # full 4-DLTS
            'high_voltage': 5000.0,
            'pulse': 0.0,
        },
        {
            'channels': [0, 0, 1, 1],
            'time_data': [200, 202, 210, 212],
            'start_counter': [11],
            'valid_event': [False],  # invalid → multi-pair recovery (2 x-hits)
            'high_voltage': 5000.0,
            'pulse': 0.0,
        },
    ]

    hit_table, _ = raw_data_workflow.extract_surface_concept_hits(sequence_records, detector_limit_cm=10.0)

    assert 'parent_pulse_length' in hit_table.columns
    # Full 4-DLTS hit: parent pulse length = 4.
    full = hit_table[hit_table['dlts'] == 4]
    assert full['parent_pulse_length'].tolist() == [4]
    # Two 2-DLTS partials: both came from a parent length-4 chunk.
    partial = hit_table[hit_table['dlts'] == 2]
    assert partial['parent_pulse_length'].tolist() == [4, 4]


def test_plot_peak_chunk_length_distribution_groups_partial_hits_by_parent_length():
    """The per-peak chunk-length distribution helper must return a Figure
    when it has data, and skip silently otherwise."""
    import matplotlib

    matplotlib.use('Agg')

    hit_table = pd.DataFrame(
        {
            'tof (ns)': [400.0, 405.0, 600.0, 600.5, 600.0],
            'mc (Da)': [27.0, 27.05, 1.0, 1.05, 27.0],
            'x_det (cm)': [0.1, 0.0, 0.0, 0.0, 0.2],
            'y_det (cm)': [0.0, 0.05, 0.0, 0.0, 0.0],
            'dlts': [4, 2, 2, 2, 2],
            'detector_axis': ['xy', 'x', 'y', 'y', 'x'],
            'in_detector': [True, True, True, True, True],
            'parent_pulse_length': [4, 4, 8, 8, 8],
        }
    )
    windows = [{'label': 'Al+', 'min': 380.0, 'max': 420.0}]

    fig = raw_data_workflow.plot_peak_chunk_length_distribution(
        hit_table,
        windows,
        signal_kind='tof',
        only_in_detector=True,
    )
    assert fig is not None
    fig.canvas.draw()
    matplotlib.pyplot.close(fig)


def test_plot_peak_helpers_cap_figure_height_to_safe_bounds_for_many_peaks():
    """Long peak lists must NOT produce figures whose pixel dimensions
    exceed matplotlib's 2^16 hard limit. With ``constrained_layout=True``
    + ``bbox_inches='tight'`` the previous code blew up to 46 M-px figures
    on user input; we now cap each helper's height to a safe bound."""
    import matplotlib

    matplotlib.use('Agg')

    # 60 peaks would have produced figsize=(11, 180) — a 60 in × 300 dpi
    # PNG → 18 000 px height; combined with bbox_inches='tight' resampling
    # this could escalate to >65 535 px. The cap brings it back below 24 in.
    windows = [{'label': f'P{n}', 'min': 0.0 + n, 'max': 1.0 + n} for n in range(60)]
    hit_table = pd.DataFrame(
        {
            'tof (ns)': [0.5, 1.5, 2.5],
            'mc (Da)': [0.5, 1.5, 2.5],
            'x_det (cm)': [0.0, 0.1, -0.1],
            'y_det (cm)': [0.0, 0.0, 0.0],
            'dlts': [4, 2, 4],
            'detector_axis': ['xy', 'x', 'xy'],
            'in_detector': [True, True, True],
            'parent_pulse_length': [4, 4, 4],
        }
    )

    fig_chunk = raw_data_workflow.plot_peak_chunk_length_distribution(
        hit_table,
        windows,
        signal_kind='tof',
        only_in_detector=True,
    )
    assert fig_chunk is not None
    chunk_h = fig_chunk.get_size_inches()[1]
    assert chunk_h <= 20.0 + 1e-6, (
        f"plot_peak_chunk_length_distribution height ({chunk_h} in) must be "
        "capped at 20 in to stay below the 2^16 px hard limit at 300 dpi"
    )
    matplotlib.pyplot.close(fig_chunk)

    fig_det = raw_data_workflow.plot_peak_detector_diagnostics(
        hit_table,
        windows,
        signal_kind='tof',
        only_in_detector=True,
        detector_limit_cm=4.0,
    )
    assert fig_det is not None
    det_h = fig_det.get_size_inches()[1]
    assert det_h <= 24.0 + 1e-6, (
        f"plot_peak_detector_diagnostics height ({det_h} in) must be "
        "capped at 24 in to stay below the 2^16 px hard limit at 300 dpi"
    )
    matplotlib.pyplot.close(fig_det)


def test_plot_peak_detector_diagnostics_returns_figure_for_each_peak():
    """The per-peak detector diagnostic figure has 1 row per peak × 3 columns
    (2D FDM + 1D x + 1D y). Returns None when no peaks/data."""
    import matplotlib

    matplotlib.use('Agg')

    hit_table = pd.DataFrame(
        {
            'tof (ns)': [400.0, 405.0],
            'mc (Da)': [27.0, 27.05],
            'x_det (cm)': [0.1, 0.2],
            'y_det (cm)': [0.0, 0.0],
            'dlts': [4, 2],
            'detector_axis': ['xy', 'x'],
            'in_detector': [True, True],
            'parent_pulse_length': [4, 4],
        }
    )
    windows = [
        {'label': 'Al+', 'min': 380.0, 'max': 420.0},
        {'label': 'Al2+', 'min': 100.0, 'max': 200.0},  # empty peak
    ]

    fig = raw_data_workflow.plot_peak_detector_diagnostics(
        hit_table,
        windows,
        signal_kind='tof',
        only_in_detector=True,
        detector_limit_cm=4.0,
    )
    assert fig is not None
    # 2 peaks × 3 panels = 6 axes total.
    assert len(fig.axes) == 6
    matplotlib.pyplot.close(fig)


def test_load_detector_constants_returns_self_consistent_geometry():
    """The new config-driven detector-constants helper must return a
    self-consistent dict for SC and RoentDek even when the caller supplies
    only a subset of overrides."""
    sc = raw_data_workflow.load_detector_constants('surface_concept', None)
    assert sc['detector_bins'] == 4900
    assert sc['detector_width_mm'] == 80.0
    # xy_factor is derived from width / bins * binning_factor — must match
    # the historical XY_FACTOR even after the refactor.
    assert sc['xy_factor'] == raw_data_workflow.XY_FACTOR

    ro = raw_data_workflow.load_detector_constants('roentdek', None)
    assert ro['detector_bins'] == 4900
    assert ro['detector_width_mm'] == 80.0
    # xy_factor self-consistency.
    assert ro['xy_factor'] == 80.0 / 4900.0 * 2.0

    # Per-rig override via a plain dict (mimicking the loaded config.toml).
    overridden = raw_data_workflow.load_detector_constants(
        'surface_concept',
        {
            'sc_detector_width_mm': 120.0,
            'sc_detector_bins': 6000,
            'sc_detector_binning_factor': 4,
            'sc_detector_limit_cm': 5.5,
        },
    )
    assert overridden['detector_width_mm'] == 120.0
    assert overridden['detector_bins'] == 6000
    assert overridden['binning_factor'] == 4
    assert overridden['detector_limit_cm'] == 5.5
    # xy_factor must reflect the new geometry, not the SC default.
    assert overridden['xy_factor'] == 120.0 / 6000.0 * 4.0


def test_surface_concept_recovery_diagnostics_include_unrecoverable_rows():
    sequence_records = [
        {
            'channels': [0, 1],
            'time_data': [100, 110],
            'start_counter': [10],
            'valid_event': [False],
            'high_voltage': 5000.0,
            'pulse': 0.0,
        },
        {
            'channels': [0],
            'time_data': [120],
            'start_counter': [11],
            'valid_event': [False],
            'high_voltage': 5000.0,
            'pulse': 0.0,
        },
    ]

    diagnostics = raw_data_workflow.build_surface_concept_recovery_diagnostics(sequence_records, detector_limit_cm=10.0)

    assert '2 DLTS in detector' in diagnostics['status'].tolist()
    assert 'unrecoverable' in diagnostics['status'].tolist()


def test_compute_tof_segment_drift_returns_peak_positions():
    dataframe = pd.DataFrame(
        {
            'tof (ns)': np.concatenate(
                [
                    np.linspace(98.0, 102.0, 60),
                    np.linspace(99.0, 103.0, 60),
                ]
            )
        }
    )

    drift = raw_data_workflow.compute_tof_segment_drift(
        dataframe,
        windows=[{'label': 'peak', 'min': 97.0, 'max': 104.0}],
        num_segments=4,
        max_value=110.0,
    )

    assert not drift.empty
    assert set(drift['peak_label']) == {'peak'}


# ---------------------------------------------------------------------------
# Combinatorial per-pulse hit recovery
# ---------------------------------------------------------------------------


def _record(channels, times, *, start=10, hv=5000.0, pulse=0.0):
    """Build a single ``find_consecutive_sequences`` record for testing."""
    return {
        'channels': list(channels),
        'time_data': list(times),
        'start_counter': [int(start)],
        'high_voltage': float(hv),
        'pulse': float(pulse),
        # valid_event is unused by the combinatorial recovery — it never
        # consults the legacy chunked validity flag.
        'valid_event': [False],
    }


def test_combinatorial_recovery_example1_two_x_pairs_when_both_in_detector():
    """Example 1 — chunk [0, 0, 1, 1] with two ions whose reconstructed
    det_x both fit the detector. Greedy must emit *two* x-axis 2-DLTS hits,
    pairing each ch=0 with the ch=1 partner that gives a valid det_x."""
    record = _record(
        channels=[0, 0, 1, 1],
        # First ion: (t=100, t=110) → det_x small.
        # Second ion: (t=200, t=210) → det_x small.
        times=[100, 200, 110, 210],
    )
    hits, candidates = raw_data_workflow.extract_valid_hits_combinatorial(
        record,
        peak_windows=None,  # no signal gate → all in-detector pairs are valid
        signal_kind='tof',
        detector_limit_cm=10.0,
        mode='greedy',
    )

    # Four candidate x-pairs were considered: (0,2), (0,3), (1,2), (1,3).
    x_candidates = [c for c in candidates if c['detector_axis'] == 'x']
    assert len(x_candidates) == 4
    # Two index-disjoint x-pair hits must come out.
    axes = sorted(h['detector_axis'] for h in hits)
    assert axes == ['x', 'x']
    used = set().union(*(h['used_indices'] for h in hits))
    assert used == {0, 1, 2, 3}  # every timestamp claimed exactly once


def test_combinatorial_recovery_example2_three_y_candidates_picks_only_geometric_one():
    """Example 2 — chunk [2, 3, 2, 2] (sorted as [2, 2, 2, 3]). Three
    candidate y-pairs against the lone ch=3. Two of them land *outside*
    the detector and must be dropped. The remaining single in-detector
    pair is emitted regardless of whether its tof falls in the user's
    peak window — peak windows now filter at the per-peak yield helper,
    not at recovery time, so the noise baseline survives in the hit table."""
    # Times chosen so:
    #   pair (idx_ch2=0, idx_ch3=3): det_y inside detector, tof in peak → emitted
    #   pair (idx_ch2=1, idx_ch3=3): det_y outside detector              → invalid
    #   pair (idx_ch2=2, idx_ch3=3): det_y outside detector              → invalid
    t_ch2_a = 10000
    t_ch2_b = 100  # |Δ| with t_ch3=11000 = 10900 bins → det_y far outside ±4 cm
    t_ch2_c = 50000  # |Δ| with t_ch3=11000 = 39000 bins → det_y way outside
    t_ch3 = 11000
    record = _record(
        channels=[2, 2, 2, 3],
        times=[t_ch2_a, t_ch2_b, t_ch2_c, t_ch3],
    )
    peak_window = [{'label': 'Al+', 'min': 280.0, 'max': 300.0}]
    hits, candidates = raw_data_workflow.extract_valid_hits_combinatorial(
        record,
        peak_windows=peak_window,
        signal_kind='tof',
        detector_limit_cm=4.0,
        mode='greedy',
    )

    y_candidates = [c for c in candidates if c['detector_axis'] == 'y']
    assert len(y_candidates) == 3
    # Only one candidate passed the per-axis detector gate.
    in_det_y = [c for c in y_candidates if c['in_detector']]
    assert len(in_det_y) == 1
    # Exactly that one is emitted (validity = geometric only).
    assert len(hits) == 1
    assert hits[0]['detector_axis'] == 'y'
    assert hits[0]['used_indices'] == frozenset({0, 3})
    assert hits[0]['in_peak'] is True  # incidentally inside the peak window


def test_combinatorial_recovery_example3_two_y_pairs_when_both_valid():
    """Example 3 — chunk [2, 2, 3, 3] with two valid y-pair combinations.
    Greedy must emit two y-axis 2-DLTS hits; the four-candidate space
    is searched and the optimal index-disjoint pair selected."""
    record = _record(
        channels=[2, 2, 3, 3],
        times=[100, 200, 110, 210],
    )
    hits, candidates = raw_data_workflow.extract_valid_hits_combinatorial(
        record,
        peak_windows=None,
        signal_kind='tof',
        detector_limit_cm=10.0,
        mode='greedy',
    )
    y_candidates = [c for c in candidates if c['detector_axis'] == 'y']
    assert len(y_candidates) == 4
    axes = sorted(h['detector_axis'] for h in hits)
    assert axes == ['y', 'y']
    used = set().union(*(h['used_indices'] for h in hits))
    assert used == {0, 1, 2, 3}


def test_combinatorial_recovery_prefers_one_complete_over_two_partials():
    """A clean ``[0, 1, 2, 3]`` chunk yields three valid candidates:

    - 1 complete (dlts=4) using all four indices,
    - 1 x-partial (dlts=2) using indices {0, 1},
    - 1 y-partial (dlts=2) using indices {2, 3}.

    By raw count, 2 partials beats 1 complete. But physically a single ion
    that fired all four channels is one event. The user-facing contract is
    "always when all channels available we have to first check full hit
    possibility then partial hit if full hit is not valid" — so the recovery
    must commit to the complete.

    This regression test reproduces the bug the user observed where the
    exhaustive mode emitted 0 % "Four DLTS" because partial-decomposition
    won by count. Both greedy AND exhaustive must now emit the complete.
    """
    record = _record(
        channels=[0, 1, 2, 3],
        # Pair sums chosen so x_sum == y_sum within tolerance → complete
        # candidate is geometrically valid.
        times=[100, 110, 105, 105],  # x_sum = 210, y_sum = 210
    )
    for mode in ('greedy', 'exhaustive'):
        hits, _ = raw_data_workflow.extract_valid_hits_combinatorial(
            record,
            peak_windows=None,
            signal_kind='tof',
            detector_limit_cm=10.0,
            mode=mode,
            pair_sum_tolerance_bins=10.0,
        )
        assert len(hits) == 1, (
            f"mode={mode!r}: expected exactly 1 emitted hit (the complete), "
            f"got {len(hits)} ({[h['detector_axis'] for h in hits]})"
        )
        assert int(hits[0]['dlts']) == 4, (
            f"mode={mode!r}: emitted hit must be a 4-DLTS complete, "
            f"got dlts={hits[0]['dlts']!r} on axis {hits[0]['detector_axis']!r}"
        )


def test_combinatorial_recovery_phase2_partials_run_on_indices_left_by_completes():
    """Two-stage selection: when phase-1 commits a complete using
    {0, 1, 2, 3}, phase-2 partial selection only considers candidates whose
    indices are still free. A length-6 chunk [0, 1, 2, 3, 0, 1] has:

    - 1 valid complete using indices {0, 1, 2, 3} → locked in phase 1.
    - 1 valid x-partial using indices {4, 5} → emitted in phase 2.

    This contract MUST hold for **both** the greedy (fast) and the
    exhaustive (slow) selection mode — slow mode is two-stage too, with
    phase 1 running an exhaustive max-disjoint search on completes only,
    and phase 2 running an exhaustive max-disjoint search on the partial
    candidates whose indices don't collide with the chosen completes.
    """
    record = _record(
        channels=[0, 1, 2, 3, 0, 1],
        # First four timestamps form a coincidence-passing complete.
        # Last two are an in-detector x-pair with a different tof.
        times=[100, 110, 105, 105, 500, 510],
    )
    for mode in ('greedy', 'exhaustive'):
        hits, _ = raw_data_workflow.extract_valid_hits_combinatorial(
            record,
            peak_windows=None,
            signal_kind='tof',
            detector_limit_cm=10.0,
            mode=mode,
            pair_sum_tolerance_bins=10.0,
        )
        assert len(hits) == 2, f"mode={mode!r}: expected 1 complete + 1 partial, got {len(hits)} hits"
        by_dlts = sorted(int(h['dlts']) for h in hits)
        assert by_dlts == [2, 4], f"mode={mode!r}: expected dlts mix [2, 4], got {by_dlts}"
        complete = [h for h in hits if int(h['dlts']) == 4][0]
        partial = [h for h in hits if int(h['dlts']) == 2][0]
        assert complete['used_indices'] == frozenset({0, 1, 2, 3}), (
            f"mode={mode!r}: phase-1 complete must lock in indices {{0,1,2,3}}"
        )
        assert partial['used_indices'] == frozenset({4, 5}), (
            f"mode={mode!r}: phase-2 partial must run on the remaining indices {{4,5}}"
        )
        assert partial['detector_axis'] == 'x'


def test_combinatorial_recovery_two_stage_on_multi_hit_pulse_in_both_modes():
    """Two-stage on a multi-hit pulse must hold for **both** the fast
    (greedy) and the slow (exhaustive) selection modes.

    Pulse layout (length 10, two coincident ions plus a leftover x-pair):
      indices 0..3 : ion A's full event (ch 0, 1, 2, 3) with x_sum ≈ y_sum
      indices 4..7 : ion B's full event (ch 0, 1, 2, 3) with x_sum ≈ y_sum
      indices 8, 9 : a stray ch-0/ch-1 pair (in detector, no y partner)

    Phase 1 (max-disjoint on completes) MUST emit two completes (ions A and
    B). Phase 2 (max-disjoint on partials, restricted to indices {8, 9})
    MUST emit one x-partial. Total = 2 completes + 1 partial = 3 hits.

    Without the two-stage split, exhaustive would (incorrectly) decompose
    each of the two completes into 2 partials each — yielding 4 + 1 = 5
    partials, zero completes — exactly the bug the user reported in their
    slow-mode screenshot. Greedy avoids that failure mode through its
    dlts-descending ranking, but the two-stage structure must still hold so
    its phase-2 partial selection is restricted to the leftover indices.
    """
    # Ion A: ch 0 @ t=10000, ch 1 @ t=11000, ch 2 @ t=10500, ch 3 @ t=10500.
    #         x_sum = 21000, y_sum = 21000 → coincidence holds.
    # Ion B: ch 0 @ t=49000, ch 1 @ t=51000, ch 2 @ t=49500, ch 3 @ t=50500.
    #         x_sum = 100000, y_sum = 100000 → coincidence holds.
    # Stray: ch 0 @ t=200, ch 1 @ t=210 (no matching ch=2/ch=3).
    record = _record(
        channels=[0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
        times=[
            10000,
            11000,
            10500,
            10500,  # ion A indices 0..3
            49000,
            51000,
            49500,
            50500,  # ion B indices 4..7
            200,
            210,  # stray x-pair indices 8..9
        ],
    )
    for mode in ('greedy', 'exhaustive'):
        hits, _ = raw_data_workflow.extract_valid_hits_combinatorial(
            record,
            peak_windows=None,
            signal_kind='tof',
            detector_limit_cm=10.0,
            mode=mode,
            pair_sum_tolerance_bins=200.0,
        )
        by_dlts = sorted(int(h['dlts']) for h in hits)
        assert by_dlts == [2, 4, 4], (
            f"mode={mode!r}: two-stage failed — expected 2 completes + 1 partial, got dlts mix {by_dlts}"
        )
        completes = [h for h in hits if int(h['dlts']) == 4]
        partial = [h for h in hits if int(h['dlts']) == 2][0]
        locked_indices = frozenset().union(*(c['used_indices'] for c in completes))
        assert locked_indices == frozenset({0, 1, 2, 3, 4, 5, 6, 7}), (
            f"mode={mode!r}: phase-1 must lock both ions' complete-event indices"
        )
        assert partial['used_indices'] == frozenset({8, 9}), (
            f"mode={mode!r}: phase-2 partial must run on the leftover indices {{8, 9}}"
        )
        assert partial['detector_axis'] == 'x'


def test_combinatorial_recovery_complete_events_require_pair_sum_coincidence():
    """A complete-event candidate is only formed when the x-pair sum and
    y-pair sum agree within ``pair_sum_tolerance_bins``. With a tight
    tolerance, the same input that yielded two valid completes will only
    yield zero completes (and falls through to partials)."""
    # Two ions A (TOF≈350 ns) and B (TOF≈800 ns) interleaved so that, with
    # the right pairing, both can be reconstructed as full 4-DLTS events.
    # We arrange t1+t0 ≈ t3+t2 = 2 * (TDC TOF in bins) for each ion.
    # Ion A: ch0=10000, ch1=11000  (sum_x = 21000)
    #         ch2=10500, ch3=10500  (sum_y = 21000) → coincidence
    # Ion B: ch0=49000, ch1=51000  (sum_x = 100000)
    #         ch2=49500, ch3=50500  (sum_y = 100000) → coincidence
    record = _record(
        channels=[0, 0, 1, 1, 2, 2, 3, 3],
        times=[10000, 49000, 11000, 51000, 10500, 49500, 10500, 50500],
    )

    # Generous tolerance → both completes can form.
    hits_ok, _ = raw_data_workflow.extract_valid_hits_combinatorial(
        record,
        peak_windows=None,
        signal_kind='tof',
        detector_limit_cm=10.0,
        mode='greedy',
        pair_sum_tolerance_bins=200.0,
    )
    complete_hits = [h for h in hits_ok if int(h['dlts']) == 4]
    assert len(complete_hits) == 2

    # Tight tolerance → same data cannot form completes; each ion's
    # axis pairs survive as 2-DLTS partials.
    hits_tight, _ = raw_data_workflow.extract_valid_hits_combinatorial(
        record,
        peak_windows=None,
        signal_kind='tof',
        detector_limit_cm=10.0,
        mode='greedy',
        # Force a sub-bin tolerance so even our perfectly aligned synthetic
        # data fails the coincidence check (we go from |sum_x - sum_y| = 0
        # passing to ANY pair being rejected).
        pair_sum_tolerance_bins=-1.0,
    )
    assert all(int(h['dlts']) == 2 for h in hits_tight)


def test_combinatorial_recovery_axis_aware_detector_gate():
    """A 2-DLTS x-axis partial is rejected ONLY by its reconstructed det_x;
    its y coordinate (always 0.0) is not used as a rejection criterion."""
    # Times chosen so the x-pair has a huge difference → |det_x| outside.
    record = _record(
        channels=[0, 1],
        times=[100, 100000],  # Δ = 99900 → |det_x| massive → outside
    )
    hits, _ = raw_data_workflow.extract_valid_hits_combinatorial(
        record,
        peak_windows=None,
        signal_kind='tof',
        detector_limit_cm=4.0,
        mode='greedy',
    )
    # Even with no peak gate, the in-detector check rejects this single x-pair.
    assert hits == []


def test_combinatorial_recovery_greedy_and_exhaustive_agree_on_simple_inputs():
    """For a multi-pair chunk where the greedy and exhaustive modes both
    have the same valid candidate set, the emitted hit count must match.
    (The user-default greedy mode should give the same answer as the
    optional slow exhaustive mode whenever there's no ambiguity.)"""
    record = _record(
        channels=[0, 0, 1, 1],
        times=[100, 200, 110, 210],
    )
    hits_greedy, _ = raw_data_workflow.extract_valid_hits_combinatorial(
        record,
        peak_windows=None,
        signal_kind='tof',
        detector_limit_cm=10.0,
        mode='greedy',
    )
    hits_exhaust, _ = raw_data_workflow.extract_valid_hits_combinatorial(
        record,
        peak_windows=None,
        signal_kind='tof',
        detector_limit_cm=10.0,
        mode='exhaustive',
    )
    assert len(hits_greedy) == len(hits_exhaust) == 2
    assert sorted(h['detector_axis'] for h in hits_greedy) == sorted(h['detector_axis'] for h in hits_exhaust)


def test_combinatorial_recovery_noise_baseline_survives_full_pulse_range():
    """End-to-end regression for the "Full spectrum" view: when the user
    enters peak windows in TOF mode, the recovered hit table must still
    contain noise events whose tof is between (but not inside) those peak
    windows. The full-spectrum plot reads ``hit_table['tof (ns)']``
    directly, so anything filtered out at recovery time disappears from
    the noise baseline."""
    # Build a fake tdc frame containing 4 pulses, each a clean 4-DLTS event,
    # whose tofs are spaced across [50, 500] ns. Two of them fall inside
    # the user's only peak window (300, 350); the other two are noise.
    tdc = pd.DataFrame(
        {
            'start_counter': np.repeat([1, 2, 3, 4], 4).astype(np.int64),
            'channel': np.tile([0, 1, 2, 3], 4).astype(np.int64),
            # tof_4d = sum * TOF_FACTOR_NS = sum * 27.432 / 4000 ns/bin.
            # Pick sums so tof = 80, 220, 320, 480 ns respectively.
            'time_data': np.array(
                [
                    # pulse 1: tof_4d = 4*sum/4 * 27.432/4000 → sum bins = 80 / (27.432/4000) ≈ 11665
                    # We just need pair-sum coincidence (x_sum == y_sum) for the complete to form.
                    2916,
                    2916,
                    2916,
                    2917,  # tof ≈ 80 ns
                    8020,
                    8020,
                    8020,
                    8020,  # tof ≈ 220 ns
                    11665,
                    11665,
                    11665,
                    11665,  # tof ≈ 320 ns (in peak)
                    17500,
                    17500,
                    17500,
                    17500,  # tof ≈ 480 ns
                ],
                dtype=np.int64,
            ),
            'high_voltage (V)': np.full(16, 5000.0),
            'pulse_v (V)': np.full(16, 0.0),
        }
    )
    peak_windows = [{'label': 'Al+', 'min': 300.0, 'max': 350.0}]

    result = raw_data_workflow.analyze_surface_concept_tdc_frame_combinatorial(
        tdc,
        peak_windows=peak_windows,
        signal_kind='tof',
        detector_limit_cm=10.0,
        max_tof_ns=5000.0,
        mode='exhaustive',
        pair_sum_tolerance_bins=10.0,
    )
    hit_table = result['hit_table']

    # Every pulse contributed at least one geometrically-valid hit, regardless
    # of whether it was inside the peak window.
    assert len(hit_table) == 4
    tofs = sorted(hit_table['tof (ns)'].tolist())
    # Two hits are inside the peak window, two are outside (noise baseline).
    in_peak = sum(1 for t in tofs if 300.0 <= t <= 350.0)
    out_of_peak = sum(1 for t in tofs if not (300.0 <= t <= 350.0))
    assert in_peak == 1
    assert out_of_peak == 3, (
        "noise events outside the user peak window must remain in the "
        "recovered hit table — that is what feeds the Full-spectrum plot. "
        f"Got tofs={tofs}, in_peak={in_peak}, out_of_peak={out_of_peak}."
    )


def test_combinatorial_recovery_emits_geometric_hits_even_outside_peak_windows():
    """The "Full spectrum" view depends on noise events outside any peak
    window still being in the recovered hit table. So validity for emission
    is GEOMETRIC ONLY — in-detector + tof in [0, max_tof_ns]. The peak-window
    membership is computed (as ``in_peak``) but doesn't gate emission;
    downstream per-peak yield helpers do their own filter against the full
    hit table.

    This test was previously asserting the opposite (no emission when no
    pair was in peak), which made the recovered hit table contain only the
    peak regions and broke the noise-baseline view.
    """
    # Pick times so every (ch=2, ch=3) pair has small |Δ| (in detector) but
    # a low pair-sum → tof_2d ≈ 14-27 ns, well outside the (280, 300) window.
    record = _record(
        channels=[2, 3, 2, 2],
        times=[950, 1000, 900, 800],
    )
    hits, candidates = raw_data_workflow.extract_valid_hits_combinatorial(
        record,
        peak_windows=[{'label': 'Al+', 'min': 280.0, 'max': 300.0}],
        signal_kind='tof',
        detector_limit_cm=10.0,
        max_tof_ns=5000.0,
        mode='greedy',
    )
    y_candidates = [c for c in candidates if c['detector_axis'] == 'y']
    assert all(c['in_detector'] for c in y_candidates)  # geometry was fine
    assert all(c.get('in_tof_range') for c in y_candidates)  # tof_2d ≈ 14-27 ns ∈ [0, 5000]
    assert not any(c['in_peak'] for c in y_candidates)  # but outside (280, 300)
    # Even though no candidate is "in peak", at least one must be emitted
    # (max-disjoint over the geometric subset; here all three y-pairs share
    # ch=3 at index 1, so exactly one wins the index conflict).
    assert len(hits) == 1
    assert hits[0]['detector_axis'] == 'y'
    assert hits[0]['in_peak'] is False
    # max_tof gate: a pair whose tof exceeds max_tof_ns must NOT be emitted
    # even with otherwise-valid geometry.
    hits_short, _ = raw_data_workflow.extract_valid_hits_combinatorial(
        record,
        peak_windows=None,
        signal_kind='tof',
        detector_limit_cm=10.0,
        max_tof_ns=10.0,  # tighter than any pair's tof_2d
        mode='greedy',
    )
    assert hits_short == []


def test_combinatorial_recovery_tags_emitted_hits_with_parent_pulse_length_and_metadata():
    """Each emitted hit carries the parent pulse length, start_counter, and
    voltage so per-peak diagnostics can answer 'this 2-DLTS hit came from a
    length-N pulse at HV=Y.'"""
    record = _record(
        channels=[0, 1, 0, 1],
        times=[100, 110, 200, 210],
        start=42,
        hv=4321.0,
        pulse=12.5,
    )
    hits, _ = raw_data_workflow.extract_valid_hits_combinatorial(
        record,
        peak_windows=None,
        signal_kind='tof',
        detector_limit_cm=10.0,
        mode='greedy',
    )
    assert hits, "expected combinatorial recovery to emit at least one hit"
    for hit in hits:
        assert hit['parent_pulse_length'] == 4
        assert hit['start_counter'] == 42
        assert hit['high_voltage (V)'] == 4321.0
        assert hit['pulse'] == 12.5


def test_analyze_surface_concept_tdc_frame_combinatorial_returns_hit_table():
    """End-to-end smoke test for the combinatorial ``analyze_*`` orchestrator
    on a tiny tdc dataframe. Hit table must carry parent_pulse_length and
    detector_axis; candidate counts must be populated."""
    tdc = pd.DataFrame(
        {
            'start_counter': [1, 1, 1, 1],
            'channel': [0, 0, 1, 1],
            'time_data': [100, 200, 110, 210],
            'high_voltage (V)': [3000.0, 3000.0, 3000.0, 3000.0],
            'pulse_v (V)': [400.0, 400.0, 400.0, 400.0],
        }
    )
    result = raw_data_workflow.analyze_surface_concept_tdc_frame_combinatorial(
        tdc,
        peak_windows=None,
        signal_kind='tof',
        detector_limit_cm=10.0,
        mode='greedy',
    )
    hit_table = result['hit_table']
    assert {'parent_pulse_length', 'detector_axis', 'in_peak', 'in_detector'}.issubset(hit_table.columns)
    assert (hit_table['parent_pulse_length'] == 4).all()
    counts = result['candidate_counts']
    assert counts['emitted'] == len(hit_table)
    assert counts['valid'] >= counts['emitted']
    assert counts['total'] >= counts['valid']
