"""Vendor-calibrated watt requests; all SDK and serial I/O is simulated."""
from unittest.mock import Mock

import pytest

from pyccapt.control.devices import nkt_ir_power
from pyccapt.control.nkt_photonics import nktpbus_switch


@pytest.fixture
def sdk(monkeypatch):
    nkt = Mock()
    nkt.openPorts.return_value = nkt.closePorts.return_value = 0
    nkt.registerWriteU16.return_value = nkt.registerWriteU8.return_value = 0
    values = {0x37: 0, 0x02: 400000, 0x36: 10, 0x20: 1000, 0x05: 640}
    def read(port, address, register, index):
        return 0, values[register]
    nkt.registerReadU8.side_effect = read
    nkt.registerReadU16.side_effect = read
    nkt.registerReadU32.side_effect = read
    monkeypatch.setattr(nktpbus_switch, '_load_nktp_dll_module', lambda: nkt)
    monkeypatch.setattr(nktpbus_switch, '_find_oxps_address', lambda *args: 15)
    bus = Mock()
    monkeypatch.setattr(nktpbus_switch, 'switch_to_nktpbus', bus)
    return nkt, values, bus


def convert(watts=1., **options):
    return nkt_ir_power.calibrated_aom('FAKE', watts, **dict(
        max_watts=5.2, max_aom_percent=100, base_hz=400000, divider=10,
        wait=lambda seconds: None, **options))


def test_uses_firmware_conversion_instead_of_scaling_watts_by_maximum(sdk):
    nkt, values, bus = sdk
    assert convert() == (1., 640)
    nkt.registerWriteU16.assert_called_once_with('FAKE', 15, 0x20, 1000, -1)
    nkt.registerWriteU8.assert_called_once_with('FAKE', 15, 0x39, 1, -1)
    nkt.closePorts.assert_called_once_with('FAKE')
    bus.assert_called_once_with('FAKE')


@pytest.mark.parametrize('watts', [-1., 5.201, float('nan'), float('inf')])
def test_rejects_bad_requests_before_changing_protocol(sdk, watts):
    nkt, _, bus = sdk
    with pytest.raises(ValueError):
        convert(watts)
    bus.assert_not_called()
    nkt.openPorts.assert_not_called()


@pytest.mark.parametrize('register,value', [(0x37, 1), (0x02, 500000), (0x36, 20)])
def test_mode_and_timing_must_match_before_watt_write(sdk, register, value):
    nkt, values, _ = sdk
    values[register] = value
    with pytest.raises(ValueError):
        convert()
    nkt.registerWriteU16.assert_not_called()
    nkt.registerWriteU8.assert_called_once_with('FAKE', 15, 0x39, 1, -1)
    nkt.closePorts.assert_called_once()


@pytest.mark.parametrize('failure', ['write', 'read', 'refused', 'aom_limit', 'restore', 'close'])
def test_failed_conversion_never_leaves_an_sdk_handle_owned(sdk, failure):
    nkt, values, _ = sdk
    if failure == 'write':
        nkt.registerWriteU16.return_value = 1
    elif failure == 'read':
        nkt.registerReadU16.side_effect = lambda *args: (1, 0)
    elif failure == 'refused':
        values[0x20] = 500
    elif failure == 'aom_limit':
        values[0x05] = 4001
    elif failure == 'restore':
        nkt.registerWriteU8.return_value = 1
    else:
        nkt.closePorts.return_value = 1
    with pytest.raises((ValueError, nktpbus_switch.NKTPSwitchError)):
        convert()
    nkt.closePorts.assert_called_once_with('FAKE')


def test_missing_sdk_does_not_switch_the_laser(sdk, monkeypatch):
    _, _, bus = sdk
    monkeypatch.setattr(nktpbus_switch, '_load_nktp_dll_module', Mock(side_effect=RuntimeError('SDK unavailable')))
    with pytest.raises(RuntimeError, match='SDK unavailable'):
        convert()
    bus.assert_not_called()


def test_partial_sdk_open_failure_is_closed(sdk):
    nkt, _, _ = sdk
    nkt.openPorts.return_value = 1
    with pytest.raises(nktpbus_switch.NKTPSwitchError):
        convert()
    nkt.closePorts.assert_called_once_with('FAKE')
