"""Startup status parsing with real boot text, without serial hardware."""
from unittest.mock import Mock

import pytest

from pyccapt.control.nkt_photonics.origamiClassCLI import origClass


def device_with_reply(reply):
    device = origClass('FAKE')
    device._query = Mock(return_value=reply)
    return device


@pytest.mark.parametrize('code', [0, 1, 3, 5, 9, 17, 33, 65, 129, 255])
def test_named_status_reply_retains_complete_integer_range(code):
    device = device_with_reply(f'ly_oxp2_dev_status {code}')
    assert device.StatusRead(timeout=.5) == f'ly_oxp2_dev_status {code}'
    device._query.assert_called_once_with('ly_oxp2_dev_status?', timeout=.5)


def test_status_accepts_attached_timing_board_boot_banner():
    device = device_with_reply(
        'Timing Board S/N: 0155-C971-8127-529C; MCU T/B: R-007 11; '
        'FPGA T/B: R-007 11\nly_oxp2_dev_status 17')
    assert device.StatusRead() == 'ly_oxp2_dev_status 17'


def test_status_ignores_query_echo_and_unrelated_numeric_help_lines():
    device = device_with_reply(
        'ly_oxp2_dev_status?\r\n'
        'Timing board firmware 11, serial 0155\r\n'
        'e_freq=4 --> 400000 Hz\r\n'
        '  ly_oxp2_dev_status\t33  \r\n'
        'Help: 1 Listen, 3 Standby, 6 emitting\r\n')
    assert device.StatusRead() == 'ly_oxp2_dev_status 33'


@pytest.mark.parametrize('reply', [
    '', '33', 'ly_oxp2_dev_status?',
    'ly_oxp2_dev_status?\nFirmware version 33',
    'ly_oxp2_dev_status', 'ly_oxp2_dev_status -1',
    'ly_oxp2_dev_status 256', 'ly_oxp2_dev_status 999',
    'ly_oxp2_dev_status 17.0', 'ly_oxp2_dev_status 1e1',
    'ly_oxp2_dev_status=33', 'ly_oxp2_dev_status 33 trailing text',
    'Help: ly_oxp2_dev_status 33',
    'ly_oxp2_dev_status 9\nly_oxp2_dev_status 9',
    'ly_oxp2_dev_status 9\nly_oxp2_dev_status 33',
    'ly_oxp2_dev_status 9\nly_oxp2_dev_status 999',
    'ly_oxp2_dev_status 9\nly_oxp2_dev_status 9.0',
])
def test_missing_malformed_or_multiple_status_lines_are_not_connection_evidence(reply):
    with pytest.raises(ValueError, match='Unrecognised laser status'):
        device_with_reply(reply).StatusRead()
