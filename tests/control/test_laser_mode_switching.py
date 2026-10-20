"""Protocol handover regressions using exclusive simulated serial ports."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pyccapt.control.nkt_photonics import nktpbus_switch, origamiClassCLI


class SilentPort:
    def __init__(self, owner, **options):
        if owner.opened:
            raise OSError('port already owned')
        owner.opened = True
        self.owner = owner
        self.options = options
        self.commands = []
        self.closed = False

    def write(self, command):
        self.commands.append(command)
        return len(command)

    def flush(self):
        pass

    def read(self, count):
        pytest.fail('A protocol switch must not wait for a CLI reply')

    def close(self):
        self.closed = True
        self.owner.opened = False


@pytest.fixture
def serial_owner(monkeypatch):
    owner = SimpleNamespace(opened=False, ports=[])
    def open_port(**options):
        port = SilentPort(owner, **options)
        owner.ports.append(port)
        return port
    monkeypatch.setattr(origamiClassCLI.serial, 'Serial', open_port)
    return owner


def test_silent_bus_switch_flushes_exact_working_command_and_releases_port(serial_owner):
    device = origamiClassCLI.origClass('FAKE')
    assert device.open_port() == 0
    device.InterbusEnable()
    port = serial_owner.ports[0]
    assert port.commands == [b'ly_oxp2_nktpbus=1\n']
    assert port.options['baudrate'] == 38400
    assert port.closed and not serial_owner.opened
    assert device.ser is None
    # A second serial user can open the same port immediately.
    nktpbus_switch.switch_to_nktpbus('FAKE')
    assert len(serial_owner.ports) == 2
    assert not serial_owner.opened


@pytest.mark.parametrize('operation', ['write', 'flush'])
def test_bus_switch_releases_serial_handle_on_io_failure(serial_owner, operation):
    device = origamiClassCLI.origClass('FAKE')
    device.open_port()
    port = serial_owner.ports[0]
    setattr(port, operation, Mock(side_effect=OSError('USB disconnected')))
    with pytest.raises(OSError, match='USB disconnected'):
        device.InterbusEnable()
    assert device.ser is None
    assert not serial_owner.opened


def test_busy_cli_probe_reports_port_ownership_without_loading_sdk(monkeypatch):
    device = Mock(last_error=PermissionError('Access denied'))
    device.open_port.return_value = -1
    monkeypatch.setattr(origamiClassCLI, 'origClass', lambda port: device)
    with pytest.raises(nktpbus_switch.NKTPSwitchError, match='Disconnect the laser in NKT CONTROL'):
        nktpbus_switch.is_cli_responding('FAKE', raise_on_port_error=True)
    device.StatusRead.assert_not_called()
    device.close_port.assert_called_once()


@pytest.fixture
def sdk(monkeypatch):
    nkt = Mock()
    nkt.openPorts.return_value = 0
    nkt.deviceGetAllTypes.return_value = (0, b'\x00\x95' + bytes(253))
    nkt.registerWriteU8.return_value = 0
    nkt.closePorts.return_value = 0
    monkeypatch.setattr(nktpbus_switch, '_load_nktp_dll_module', lambda: nkt)
    monkeypatch.setattr(nktpbus_switch.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(nktpbus_switch, 'is_cli_responding', Mock(return_value=True))
    return nkt


def test_cli_switch_uses_documented_register_and_releases_sdk(sdk):
    assert nktpbus_switch.switch_to_cli('FAKE') == 1
    sdk.openPorts.assert_called_once_with('FAKE', 1, 0)
    sdk.registerWriteU8.assert_called_once_with('FAKE', 1, 0x39, 1, -1)
    sdk.closePorts.assert_called_once_with('FAKE')


@pytest.mark.parametrize('failure', ['open', 'scan', 'write'])
def test_sdk_releases_port_on_each_failed_switch_step(sdk, failure):
    if failure == 'open':
        sdk.openPorts.return_value = 1
    elif failure == 'scan':
        sdk.deviceGetAllTypes.return_value = (0, bytes(255))
    else:
        sdk.registerWriteU8.return_value = 1
    with pytest.raises(nktpbus_switch.NKTPSwitchError):
        nktpbus_switch.switch_to_cli('FAKE')
    sdk.closePorts.assert_called_once_with('FAKE')


def test_failed_sdk_release_is_reported_before_reopening_cli(sdk):
    sdk.closePorts.return_value = 1
    with pytest.raises(nktpbus_switch.NKTPSwitchError, match='Could not release NKTPBus port'):
        nktpbus_switch.switch_to_cli('FAKE')


def test_discovery_uses_scan_at_address_fifteen_without_native_absent_device_reads(sdk):
    types = bytearray(255)
    types[15] = 0x95
    sdk.deviceGetAllTypes.return_value = (0, bytes(types))
    sdk.deviceGetType.side_effect = AssertionError('Do not probe absent device addresses')
    assert nktpbus_switch.switch_to_cli('FAKE') == 15
    sdk.deviceGetType.assert_not_called()


@pytest.mark.parametrize('ack', [0, 6])
def test_protocol_write_closes_sdk_then_verifies_cli_even_after_reset_timeout(sdk, monkeypatch, ack):
    sdk.registerWriteU8.return_value = ack
    events = []
    def close(port):
        events.append('sdk closed')
        return 0
    sdk.closePorts.side_effect = close
    def probe(port, **options):
        assert events[0] == 'sdk closed'
        assert options['raise_on_port_error']
        events.append('cli probe')
        return len(events) >= 3  # One boot-time silence, then valid CLI.
    monkeypatch.setattr(nktpbus_switch, 'is_cli_responding', probe)
    assert nktpbus_switch.switch_to_cli('FAKE', device_address=15) == 15
    assert events == ['sdk closed', 'cli probe', 'cli probe']
    sdk.registerWriteU8.assert_called_once_with('FAKE', 15, 0x39, 1, -1)


@pytest.mark.parametrize('ack', [0, 6])
def test_timeout_or_ack_without_cli_status_is_not_success(sdk, monkeypatch, ack):
    sdk.registerWriteU8.return_value = ack
    probe = Mock(return_value=False)
    monkeypatch.setattr(nktpbus_switch, 'is_cli_responding', probe)
    with pytest.raises(nktpbus_switch.NKTPSwitchError, match='CLI was not verified'):
        nktpbus_switch.switch_to_cli('FAKE', device_address=15)
    assert probe.call_count == 3
    sdk.registerWriteU8.assert_called_once()
    sdk.closePorts.assert_called_once()


def test_non_timeout_protocol_error_is_not_ignored(sdk, monkeypatch):
    sdk.registerWriteU8.return_value = 4  # RegResultNacked, not a reset timeout.
    probe = Mock(return_value=True)
    monkeypatch.setattr(nktpbus_switch, 'is_cli_responding', probe)
    with pytest.raises(nktpbus_switch.NKTPSwitchError, match='registerWriteU8 failed'):
        nktpbus_switch.switch_to_cli('FAKE')
    probe.assert_not_called()
    sdk.closePorts.assert_called_once()


def test_transient_port_open_failure_is_retried_only_after_sdk_closes(sdk, monkeypatch):
    probe = Mock(side_effect=[nktpbus_switch.NKTPSwitchError('port settling'), True])
    monkeypatch.setattr(nktpbus_switch, 'is_cli_responding', probe)
    assert nktpbus_switch.switch_to_cli('FAKE') == 1
    assert probe.call_count == 2


def test_shutdown_during_cli_settle_cancels_probe(sdk, monkeypatch):
    probe = Mock()
    monkeypatch.setattr(nktpbus_switch, 'is_cli_responding', probe)
    with pytest.raises(nktpbus_switch.NKTPSwitchError, match='cancelled'):
        nktpbus_switch.switch_to_cli('FAKE', wait=lambda delay: None, cancelled=lambda: True)
    sdk.closePorts.assert_called_once()
    probe.assert_not_called()


def test_failed_sdk_import_does_not_poison_next_attempt(monkeypatch, tmp_path):
    sdk = tmp_path / 'sdk'
    example = sdk / 'Examples' / 'DLL_Example_Python' / 'NKTP_DLL.py'
    example.parent.mkdir(parents=True)
    example.write_text("raise OSError('DLL unavailable')\n", encoding='utf-8')
    monkeypatch.setenv('NKTP_SDK_PATH', str(sdk))
    monkeypatch.delitem(sys.modules, '_pyccapt_NKTP_DLL', raising=False)
    with pytest.raises(nktpbus_switch.NKTPSwitchError, match='DLL unavailable'):
        nktpbus_switch._load_nktp_dll_module()
    assert '_pyccapt_NKTP_DLL' not in sys.modules
    example.write_text('loaded_successfully = True\n', encoding='utf-8')
    module = nktpbus_switch._load_nktp_dll_module()
    assert module.loaded_successfully
    sys.modules.pop('_pyccapt_NKTP_DLL', None)


def test_new_cli_session_clears_partial_input_and_drains_help_before_status(monkeypatch):
    class Port:
        elapsed = 0.
        pending = b''
        commands = []
        partial = b'garbage left by bus traffic'
        def write(self, command):
            self.commands.append(command)
            if command == b'\n':
                self.partial = b''
                self.pending = b'Error: Wrong command\nBasic command list for the laser\n' * 20
            else:
                assert not self.partial and not self.pending
                self.pending = b'ly_oxp2_dev_status?\r\nly_oxp2_dev_status 17\n'
        def flush(self): pass
        def reset_input_buffer(self): self.pending = b''
        @property
        def in_waiting(self): return min(200, len(self.pending))
        def read(self, count):
            self.elapsed += .05
            data, self.pending = self.pending[:count], self.pending[count:]
            return data
        def close(self): pass
    port = Port()
    monkeypatch.setattr(origamiClassCLI.serial, 'Serial', lambda **options: port)
    monkeypatch.setattr(origamiClassCLI, 'time', SimpleNamespace(monotonic=lambda: port.elapsed))
    device = origamiClassCLI.origClass('FAKE')
    assert device.open_port() == 0
    assert device.StatusRead() == 'ly_oxp2_dev_status 17'
    assert device.StatusRead() == 'ly_oxp2_dev_status 17'
    assert port.commands == [b'\n', b'ly_oxp2_dev_status?\r\n', b'ly_oxp2_dev_status?\r\n']
    device.close_port()
