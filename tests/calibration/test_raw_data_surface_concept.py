import numpy as np

from pyccapt.calibration.data_tools import raw_data_surface_concept


def test_find_consecutive_sequences_requires_full_0123_pattern_for_valid_four_channel_hit():
    start_counter = np.array([1, 1, 1, 1], dtype=np.int64)
    channel = np.array([0, 0, 2, 3], dtype=np.int64)
    time_data = np.array([10, 20, 30, 40], dtype=np.int64)
    high_voltage = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    pulse = np.array([0.0, 0.0, 0.0, 0.0])

    result = raw_data_surface_concept.find_consecutive_sequences(
        start_counter,
        channel,
        time_data,
        high_voltage,
        pulse,
        print_stats=False,
    )

    assert len(result) == 1
    assert result[0]['valid_event'] == [False]


def test_find_consecutive_sequences_handles_last_sequence_without_reusing_previous_flags():
    start_counter = np.array([1, 1, 1, 1], dtype=np.int64)
    channel = np.array([0, 1, 2, 3], dtype=np.int64)
    time_data = np.array([10, 20, 30, 40], dtype=np.int64)
    high_voltage = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    pulse = np.array([0.0, 0.0, 0.0, 0.0])

    result = raw_data_surface_concept.find_consecutive_sequences(
        start_counter,
        channel,
        time_data,
        high_voltage,
        pulse,
        print_stats=False,
    )

    assert len(result) == 1
    assert result[0]['channels'] == [0, 1, 2, 3]
    assert result[0]['valid_event'] == [True]
