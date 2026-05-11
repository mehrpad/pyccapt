from tqdm.auto import tqdm


def _sort_sequence_by_channel(start_counter_values, channel_values, time_values):
    indexed_channels = sorted(enumerate(channel_values), key=lambda item: item[1])
    sorted_indices = [index for index, _ in indexed_channels]
    sorted_channels = [channel_values[index] for index in sorted_indices]
    sorted_time = [time_values[index] for index in sorted_indices]
    sorted_start_counter = [start_counter_values[index] for index in sorted_indices]
    return sorted_channels, sorted_time, sorted_start_counter


def _build_valid_event_flags(sorted_channels):
    length = len(sorted_channels)
    if length < 4:
        return [False]
    if length == 4:
        return [sorted_channels == [0, 1, 2, 3]]

    valid_event = []
    for index_valid_event in range((length + 3) // 4):
        chunk = sorted_channels[index_valid_event * 4 : (index_valid_event + 1) * 4]
        valid_event.append(chunk == [0, 1, 2, 3])
    return valid_event


def _normalize_sequence(current_sequence, ch, time):
    sc = current_sequence.copy()
    length = len(current_sequence)
    if length <= 4:
        sorted_channels, sorted_time, sorted_start_counter = _sort_sequence_by_channel(sc, ch, time)
        valid_event = _build_valid_event_flags(sorted_channels)
        return sorted_channels, sorted_time, sorted_start_counter, valid_event

    ch_sorted = []
    index = []
    k = 0
    j = 0
    while len(ch) != len(ch_sorted):
        if j == len(ch):
            j = 0
            k = k + 1
            if k == 4:
                k = 0

        if ch[j] == k:
            if j not in index:
                index.append(j)
                ch_sorted.append(ch[index[-1]])
                k = k + 1
                if k == 4:
                    k = 0
                j = 0
                continue
        j = j + 1

    sorted_time = [time[idx] for idx in index]
    sorted_start_counter = [sc[idx] for idx in index]
    valid_event = _build_valid_event_flags(ch_sorted)
    return ch_sorted, sorted_time, sorted_start_counter, valid_event


def find_consecutive_sequences_seperatly(start_counter, channel, time_data, high_voltage, pulse):
    """ "
    find the consecutive sequences of the start counter and the corresponding channels
    Args:
        start_counter: list of start counter values
        channel: list of channel values
        time_data: list of time data values
        high_voltage: list of high voltage values
        pulse: list of pulse values
    Return:
        result_4: list of dictionaries containing the valid sequences of 4 channels
        result_4_invalid: list of dictionaries containing the invalid sequences of 4 channels
        result_3_invalid: list of dictionaries containing the invalid sequences of 3 channels
        result_2_invalid: list of dictionaries containing the invalid sequences of 2 channels
        result_1_invalid: list of dictionaries containing the invalid sequences of 1 channels
        result_other_odd: list of dictionaries containing the sequences of odd length
        result_other_even: list of dictionaries containing the sequences of even length
    """
    result_4 = []
    result_4_invalid = []
    result_3_invalid = []
    result_2_invalid = []
    result_1_invalid = []
    result_other_odd = []
    result_other_even = []

    current_sequence = []
    ch = []
    time = []
    current_start = 0
    current_high_voltage = None
    current_pulse = None

    # for i, value in enumerate(start_counter):
    for i, value in tqdm(enumerate(start_counter), desc="Processing", total=len(start_counter)):
        hv = high_voltage[i]
        pulse_c = pulse[i]
        if i == 0:
            current_sequence.append(value)
            ch.append(channel[i])
            time.append(time_data[i])
            current_high_voltage = hv
            current_pulse = pulse_c
            continue
        if current_sequence[-1] == value:
            current_sequence.append(value)
            ch.append(channel[i])
            time.append(time_data[i])
        else:
            length = len(current_sequence)
            ch, time, sc, valid_event = _normalize_sequence(current_sequence, ch, time)

            if length == 4:
                if valid_event[0]:
                    result_4.append(
                        {
                            'channels': ch,
                            'time_data': time,
                            'start_counter': sc,
                            'valid_event': valid_event,
                            'high_voltage': current_high_voltage,
                            'pulse': current_pulse,
                            'indices': (current_start, i - 1),
                            'length': length,
                        }
                    )
                else:
                    result_4_invalid.append(
                        {
                            'channels': ch,
                            'time_data': time,
                            'start_counter': sc,
                            'valid_event': valid_event,
                            'high_voltage': current_high_voltage,
                            'pulse': current_pulse,
                            'indices': (current_start, i - 1),
                            'length': length,
                        }
                    )
            elif length == 3:
                result_3_invalid.append(
                    {
                        'channels': ch,
                        'time_data': time,
                        'start_counter': sc,
                        'valid_event': valid_event,
                        'high_voltage': current_high_voltage,
                        'pulse': current_pulse,
                        'indices': (current_start, i - 1),
                        'length': length,
                    }
                )
            elif length == 2:
                result_2_invalid.append(
                    {
                        'channels': ch,
                        'time_data': time,
                        'start_counter': sc,
                        'valid_event': valid_event,
                        'high_voltage': current_high_voltage,
                        'pulse': current_pulse,
                        'indices': (current_start, i - 1),
                        'length': length,
                    }
                )
            elif length == 1:
                result_1_invalid.append(
                    {
                        'channels': ch,
                        'time_data': time,
                        'start_counter': sc,
                        'valid_event': valid_event,
                        'high_voltage': current_high_voltage,
                        'pulse': current_pulse,
                        'indices': (current_start, i - 1),
                        'length': length,
                    }
                )
            else:
                if length % 4 == 0:
                    result_other_even.append(
                        {
                            'channels': ch,
                            'time_data': time,
                            'start_counter': sc,
                            'valid_event': valid_event,
                            'high_voltage': current_high_voltage,
                            'pulse': current_pulse,
                            'indices': (current_start, i - 1),
                            'length': length,
                        }
                    )

                else:
                    result_other_odd.append(
                        {
                            'channels': ch,
                            'time_data': time,
                            'start_counter': sc,
                            'valid_event': valid_event,
                            'high_voltage': current_high_voltage,
                            'pulse': current_pulse,
                            'indices': (current_start, i - 1),
                            'length': length,
                        }
                    )

            current_sequence = [value]
            ch = [channel[i]]
            time = [time_data[i]]
            current_start = i
            current_high_voltage = hv
            current_pulse = pulse_c

        # print(f"Processing: {i} / {len(start_counter)}")

    # Handle the last sequence
    length = len(current_sequence)
    ch, time, sc, valid_event = _normalize_sequence(current_sequence, ch, time)
    if length == 4:
        if valid_event[0]:
            result_4.append(
                {
                    'channels': ch,
                    'time_data': time,
                    'start_counter': sc,
                    'valid_event': valid_event,
                    'high_voltage': current_high_voltage,
                    'pulse': current_pulse,
                    'indices': (current_start, len(start_counter) - 1),
                    'length': length,
                }
            )
        else:
            result_4_invalid.append(
                {
                    'channels': ch,
                    'time_data': time,
                    'start_counter': sc,
                    'valid_event': valid_event,
                    'high_voltage': current_high_voltage,
                    'pulse': current_pulse,
                    'indices': (current_start, len(start_counter) - 1),
                    'length': length,
                }
            )
    elif length == 3:
        result_3_invalid.append(
            {
                'channels': ch,
                'time_data': time,
                'start_counter': sc,
                'valid_event': valid_event,
                'high_voltage': current_high_voltage,
                'pulse': current_pulse,
                'indices': (current_start, len(start_counter) - 1),
                'length': length,
            }
        )
    elif length == 2:
        result_2_invalid.append(
            {
                'channels': ch,
                'time_data': time,
                'start_counter': sc,
                'valid_event': valid_event,
                'high_voltage': current_high_voltage,
                'pulse': current_pulse,
                'indices': (current_start, len(start_counter) - 1),
                'length': length,
            }
        )
    elif length == 1:
        result_1_invalid.append(
            {
                'channels': ch,
                'time_data': time,
                'start_counter': sc,
                'valid_event': valid_event,
                'high_voltage': current_high_voltage,
                'pulse': current_pulse,
                'indices': (current_start, len(start_counter) - 1),
                'length': length,
            }
        )
    else:
        if length % 4 == 0:
            result_other_even.append(
                {
                    'channels': ch,
                    'time_data': time,
                    'start_counter': sc,
                    'valid_event': valid_event,
                    'high_voltage': current_high_voltage,
                    'pulse': current_pulse,
                    'indices': (current_start, len(start_counter) - 1),
                    'length': length,
                }
            )
        else:
            result_other_odd.append(
                {
                    'channels': ch,
                    'time_data': time,
                    'start_counter': sc,
                    'valid_event': valid_event,
                    'high_voltage': current_high_voltage,
                    'pulse': current_pulse,
                    'indices': (current_start, len(start_counter) - 1),
                    'length': length,
                }
            )

    lenght_result_4 = len(result_4) * 4
    length_result_4_invalid = len(result_4_invalid) * 4
    lenght_result_3 = len(result_3_invalid) * 3
    lenght_result_2 = len(result_2_invalid) * 2
    lenght_result_1 = len(result_1_invalid) * 1
    lenght_result_other_odd = sum(item['length'] for item in result_other_odd)
    lenght_result_other_even = sum(item['length'] for item in result_other_even)

    print(f"Length of 4 channel: {lenght_result_4 / 4}, {lenght_result_4 / len(start_counter) * 100} %")
    print(
        f"Length of 4 channel (invalid): {length_result_4_invalid / 4}, {length_result_4_invalid / len(start_counter) * 100} %"
    )
    print(f"Length of 3 channel: {lenght_result_3 / 3}, {lenght_result_3 / len(start_counter) * 100} %")
    print(f"Length of 2 channel: {lenght_result_2 / 2}, {lenght_result_2 / len(start_counter) * 100} %")
    print(f"Length of 1 channel: {lenght_result_1}, {lenght_result_1 / len(start_counter) * 100} %")
    print(
        f"Length of groups of four channel (multihit): {lenght_result_other_even}, {lenght_result_other_even / len(start_counter) * 100} %"
    )
    print(
        f"Length of not group of four channel (multihit): {lenght_result_other_odd}, {lenght_result_other_odd / len(start_counter) * 100} %"
    )

    # Check the conditions
    total_length = (
        lenght_result_4
        + length_result_4_invalid
        + lenght_result_3
        + lenght_result_2
        + lenght_result_1
        + lenght_result_other_odd
        + lenght_result_other_even
    )
    assert total_length == len(start_counter), "The total length of the sequences is not equal to the length of the array"
    print(f"Total length: {total_length}")

    return (
        result_4,
        result_4_invalid,
        result_3_invalid,
        result_2_invalid,
        result_1_invalid,
        result_other_odd,
        result_other_even,
    )


def find_consecutive_sequences(start_counter, channel, time_data, high_voltage, pulse, print_stats=False):
    """ "
        Find the consecutive sequences of the start counter and the corresponding channels
    Args:
        start_counter: list of start counter values
        channel: list of channel values
        time_data: list of time data values
        high_voltage: list of high voltage values
        pulse: list of pulse values
        print_stats: bool, print the statistics of the sequences
    Return:
        result: list of dictionaries containing the sequences
    """
    result = []
    current_sequence = []
    ch = []
    time = []
    current_start = 0
    current_high_voltage = None
    current_pulse = None
    # for i, value in enumerate(start_counter):
    for i, value in tqdm(enumerate(start_counter), desc="Processing", total=len(start_counter)):
        hv = high_voltage[i]
        pulse_c = pulse[i]
        if i == 0:
            current_sequence.append(value)
            ch.append(channel[i])
            time.append(time_data[i])
            current_high_voltage = hv
            current_pulse = pulse_c
            continue
        if current_sequence[-1] == value:
            current_sequence.append(value)
            ch.append(channel[i])
            time.append(time_data[i])
        else:
            length = len(current_sequence)
            ch, time, sc, valid_event = _normalize_sequence(current_sequence, ch, time)

            result.append(
                {
                    'channels': ch,
                    'time_data': time,
                    'start_counter': sc,
                    'valid_event': valid_event,
                    'high_voltage': current_high_voltage,
                    'pulse': current_pulse,
                    'indices': (current_start, i - 1),
                    'length': length,
                }
            )

            current_sequence = [value]
            ch = [channel[i]]
            time = [time_data[i]]
            current_start = i
            current_high_voltage = hv
            current_pulse = pulse_c

    # Handle the last sequence
    length = len(current_sequence)
    ch, time, sc, valid_event = _normalize_sequence(current_sequence, ch, time)
    result.append(
        {
            'channels': ch,
            'time_data': time,
            'start_counter': sc,
            'valid_event': valid_event,
            'high_voltage': current_high_voltage,
            'pulse': current_pulse,
            'indices': (current_start, len(start_counter) - 1),
            'length': length,
        }
    )

    if print_stats:
        lenght_result_4 = len([x for x in result if x['length'] == 4])
        length_result_4_invalid = len([x for x in result if x['length'] == 4 and x['valid_event'] == [False]])
        lenght_result_3 = len([x for x in result if x['length'] == 3])
        lenght_result_2 = len([x for x in result if x['length'] == 2])
        lenght_result_1 = len([x for x in result if x['length'] == 1])
        # lenght_result_other_even = len([x for x in result if x['length'] > 4 and x['length'] % 4 == 0])
        # lenght_result_other_odd = len([x for x in result if x['length'] > 4 and x['length'] % 4 != 0])
        lenght_result_other_even = 0
        for i in range(len(result)):
            if result[i]['length'] % 4 == 0 and result[i]['length'] > 4:
                lenght_result_other_even += result[i]['length']
        lenght_result_other_odd = 0
        for i in range(len(result)):
            if result[i]['length'] % 4 != 0 and result[i]['length'] > 4:
                lenght_result_other_odd += result[i]['length']
        # length of valid events if results that has lenth bigger than 4
        lenght_of_valid_even_events = len([x for x in result if x['length'] > 4 and x['valid_event'] == [True]])
        print(f"Length of 4 channel: {lenght_result_4}, {lenght_result_4 * 4 / len(start_counter) * 100} %")
        print(
            f"Length of 4 channel (invalid): {length_result_4_invalid}, {length_result_4_invalid * 4 / len(start_counter) * 100} %"
        )
        print(f"Length of 3 channel: {lenght_result_3}, {lenght_result_3 * 3 / len(start_counter) * 100} %")
        print(f"Length of 2 channel: {lenght_result_2}, {lenght_result_2 * 2 / len(start_counter) * 100} %")
        print(f"Length of 1 channel: {lenght_result_1}, {lenght_result_1 * 1 / len(start_counter) * 100} %")
        print(
            f"Length of groups of four channel (multi hit): {lenght_result_other_even}, {lenght_result_other_even / len(start_counter) * 100} %"
        )
        print(
            f"Length of not group of four and one less than 4 channel (multi hit): {lenght_result_other_odd}, {lenght_result_other_odd / len(start_counter) * 100} %"
        )

        # Check the conditions
        total_length = (
            lenght_result_4 * 4
            + lenght_result_3 * 3
            + lenght_result_2 * 2
            + lenght_result_1 * 1
            + lenght_result_other_odd
            + lenght_result_other_even
        )
        assert total_length == len(start_counter), "The total length of the sequences is not equal to the length of the array"
        print(f"Total length: {total_length}")

    return result


def find_nth_max_repeated_indices(nums, n):
    """
    Find the start/end indices of the ``n``-th longest repeated run (1-based).

    Args:
        nums:
        n: 1 returns the longest run, 2 the second-longest, and so on.
    Returns:

    """
    if n < 1:
        raise ValueError("n must be >= 1")

    while n > 0:
        max_count = 0
        max_number = None
        start_index = None
        end_index = None

        current_count = 0
        current_number = None

        for i, num in enumerate(nums):
            if num != -1:
                if num != current_number:
                    current_number = num
                    current_count = 1
                else:
                    current_count += 1

                if current_count > max_count:
                    max_count = current_count
                    max_number = current_number
                    start_index = i - current_count + 1
                    end_index = i
        n = n - 1
        if n == 0 or start_index is None:
            break
        nums[start_index : end_index + 1] = [-1] * max_count

    return start_index, end_index, max_count, max_number
