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


def test_surface_concept_recovery_extracts_two_partial_pairs_from_one_chunk():
    sequence_records = [
        {
            'channels': [0, 1, 2, 3],
            'time_data': [100, 110, 120, 130],
            'start_counter': [10],
            'valid_event': [False],
            'high_voltage': 5000.0,
            'pulse': 0.0,
        },
    ]

    hit_table, recovery = raw_data_workflow.extract_surface_concept_hits(sequence_records, detector_limit_cm=10.0)

    assert recovery['recovered_hits'] == 2
    assert recovery['one_d_hits'] == 2
    assert sorted(hit_table['detector_axis'].tolist()) == ['x', 'y']


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
