"""Startup selects Listen using the current protocol, with simulated devices."""
import os
import struct
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pyccapt.control.gui import gui_laser_control
from pyccapt.control.gui.gui_laser_control import Ui_Laser_Control
from pyccapt.control.nkt_photonics import nktpbus_switch, origamiClassCLI


@pytest.fixture
def startup_gui(monkeypatch):
    ui = Ui_Laser_Control.__new__(Ui_Laser_Control)
    ui.laser_device = None
    ui.variables = SimpleNamespace(start_flag=False)
    ui.conf = {'laser_ir_max_power_w': 5.2, 'laser_aom_max_percent': 100.}
    ui.com_port_laser = 'FAKE'
    ui._last_laser_status = None
    ui._laser_alignment_busy = Mock(return_value=False)
    ui._publish_laser_state = Mock()
    ui._request_laser_state = Mock()
    ui._mark_laser_indicators_sent = Mock()
    ui._sync_controls_from_device = Mock(return_value=True)
    ui._set_laser_disconnected_banner = Mock()
    ui._update_laser_mode_buttons = Mock()
    ui._laser_status_timer = Mock()
    ui._apply_button_locks_for_status = Mock(
        side_effect=lambda text: setattr(ui, '_last_laser_status', text))
    def close():
        if ui.laser_device is not None:
            ui.laser_device.close_port()
            ui.laser_device = None
    ui._close_laser_device = Mock(side_effect=close)
    for method in ('switch_to_cli', 'switch_to_nktpbus', 'request_cli_protocol'):
        monkeypatch.setattr(nktpbus_switch, method, Mock(side_effect=AssertionError('No protocol writes at startup')))
    return ui


@pytest.mark.parametrize('status', [9, 17, 33, 65, 129])
def test_startup_selects_listen_on_cli_without_sdk_probe_or_upward_commands(startup_gui, monkeypatch, status):
    ui = startup_gui
    cli = Mock(last_error=None)
    cli.open_port.return_value = 0
    cli.StatusRead.return_value = f'ly_oxp2_dev_status {status}'
    cli.Listen.side_effect = lambda: setattr(cli.StatusRead, 'return_value', 'ly_oxp2_dev_status 9')
    monkeypatch.setattr(origamiClassCLI, 'origClass', Mock(return_value=cli))
    native = Mock(side_effect=AssertionError('A responding CLI needs no SDK scan'))
    monkeypatch.setattr(gui_laser_control, 'NativeLaser', native)
    ui._recover_laser_startup()
    assert ui.laser_device is cli
    native.assert_not_called()
    ui._sync_controls_from_device.assert_called_once_with(initial=True, status_text='ly_oxp2_dev_status 9')
    assert cli.Listen.call_count == (0 if status == 9 else 1)
    for method in ('Standby', 'Enable', 'AOM', 'InterbusEnable'):
        getattr(cli, method).assert_not_called()


@pytest.mark.parametrize('status', [9, 17, 33, 65, 129])
def test_silent_cli_selects_listen_on_existing_bus_without_protocol_or_power_writes(startup_gui, monkeypatch, status):
    ui = startup_gui
    cli = Mock(last_error=None)
    cli.open_port.return_value = 0
    cli.StatusRead.side_effect = TimeoutError('received 0 bytes')
    monkeypatch.setattr(origamiClassCLI, 'origClass', Mock(return_value=cli))
    device = Mock(last_error=None)
    device.open_port.return_value = 0
    device.StatusRead.return_value = f'ly_oxp2_dev_status {status}'
    device.Listen.side_effect = lambda: setattr(device.StatusRead, 'return_value', 'ly_oxp2_dev_status 9')
    def create(*args, **kwargs):
        cli.close_port.assert_called_once()  # Never overlap CLI and SDK owners.
        return device
    factory = Mock(side_effect=create)
    monkeypatch.setattr(gui_laser_control, 'NativeLaser', factory)
    ui._recover_laser_startup()
    assert ui.laser_device is device
    factory.assert_called_once_with('FAKE', wavelength=None, max_watts=5.2, max_aom_percent=100.)
    ui._publish_laser_state.assert_called_once_with('ly_oxp2_dev_status 9')
    ui._sync_controls_from_device.assert_called_once_with(initial=True, status_text='ly_oxp2_dev_status 9')
    ui._set_laser_disconnected_banner.assert_called_with('')
    assert device.Listen.call_count == (0 if status == 9 else 1)
    for method in ('Standby', 'Enable', 'AOM', 'AOMEnable', 'AOMDisable', 'set_ir_power_watts'):
        getattr(device, method).assert_not_called()
    device.close_port.assert_not_called()
    assert not ui._laser_mode_switch_in_progress


@pytest.mark.parametrize('failure', ['port_busy', 'close_failed'])
def test_unreleased_cli_port_prevents_sdk_probe(startup_gui, monkeypatch, failure):
    ui = startup_gui
    cli = Mock(last_error=PermissionError('Port already owned'))
    cli.open_port.return_value = -1 if failure == 'port_busy' else 0
    cli.StatusRead.side_effect = TimeoutError('received 0 bytes')
    if failure == 'close_failed':
        cli.close_port.side_effect = OSError('Close failed')
    monkeypatch.setattr(origamiClassCLI, 'origClass', Mock(return_value=cli))
    monkeypatch.setattr(gui_laser_control, '_available_serial_ports_text', lambda: 'FAKE')
    factory = Mock(side_effect=AssertionError('Do not overlap a busy CLI port'))
    monkeypatch.setattr(gui_laser_control, 'NativeLaser', factory)
    ui._recover_laser_startup()
    factory.assert_not_called()
    assert not ui._laser_cli_query_failed
    if failure == 'close_failed':
        assert ui.laser_device is cli


@pytest.mark.parametrize('failure', ['missing_sdk', 'no_device', 'bad_status'])
def test_failed_native_detection_releases_handle_and_does_not_switch_protocol(startup_gui, monkeypatch, failure):
    ui = startup_gui
    device = Mock(last_error=ConnectionError('No OXPS response'))
    device.open_port.return_value = -1 if failure == 'no_device' else 0
    device.StatusRead.side_effect = ValueError('Malformed native status') if failure == 'bad_status' else None
    factory = Mock(side_effect=ImportError('SDK unavailable')) if failure == 'missing_sdk' else Mock(return_value=device)
    monkeypatch.setattr(gui_laser_control, 'NativeLaser', factory)
    assert not ui._open_existing_laser_native('FAKE')
    assert ui.laser_device is None
    if failure != 'missing_sdk':
        device.close_port.assert_called_once()
    assert 'No protocol change or emission was requested' in ui._set_laser_disconnected_banner.call_args.args[0]
    assert not ui._laser_mode_switch_in_progress


@pytest.mark.parametrize('blocked', ['connected', 'stopping', 'transition', 'poll', 'experiment', 'alignment'])
def test_native_startup_detection_is_guarded(startup_gui, monkeypatch, blocked):
    ui = startup_gui
    ui.laser_device = Mock() if blocked == 'connected' else None
    ui._laser_stopping = blocked == 'stopping'
    ui._laser_mode_switch_in_progress = blocked == 'transition'
    ui._laser_status_in_progress = blocked == 'poll'
    ui.variables.start_flag = blocked == 'experiment'
    ui._laser_alignment_busy.return_value = blocked == 'alignment'
    factory = Mock(side_effect=AssertionError('Startup connection is blocked'))
    monkeypatch.setattr(gui_laser_control, 'NativeLaser', factory)
    assert not ui._open_existing_laser_native('FAKE')
    factory.assert_not_called()


@pytest.mark.parametrize('fsm,status', [(1, 9), (2, 17), (3, 33), (5, 65), (6, 129)])
def test_actual_native_startup_adapter_only_requests_listen_on_existing_bus(startup_gui, monkeypatch, fsm, status):
    ui = startup_gui
    sdk = Mock()
    sdk.openPorts.return_value = sdk.closePorts.return_value = 0
    sdk.deviceGetAllTypes.return_value = (0, bytes(15) + b'\x95' + bytes(239))
    registers = {0x01: fsm, 0x67: 0, 0x32: 2, 0x35: 4, 0x36: 10,
                 0x05: 800, 0x10: 1000, 0x20: 1200, 0x39: 0}
    for kind in ('U8', 'U16', 'U32'):
        getattr(sdk, 'registerRead'+kind).side_effect = (
            lambda port, address, register, index: (0, registers[register]))
        getattr(sdk, 'registerWrite'+kind).side_effect = AssertionError('Startup must not change power or protocol')
    def listen_write(port, address, register, value, index):
        assert (register, value) == (0x30, 1)
        registers[0x01] = 1
        return 0
    sdk.registerWriteU8.side_effect = listen_write
    sdk.registerRead.return_value = (0, struct.pack('<13I', *([0]*4 + [400000] + [0]*8)))
    monkeypatch.setattr(nktpbus_switch, '_load_nktp_dll_module', lambda: sdk)
    assert ui._open_existing_laser_native('FAKE')
    device = ui.laser_device
    assert device._wavelength_index is None
    assert device.status_details['fsm'] == 1
    ui._publish_laser_state.assert_called_once_with('ly_oxp2_dev_status 9')
    sdk.openPorts.assert_called_once()
    sdk.closePorts.assert_not_called()
    assert sdk.registerWriteU8.call_count == (0 if fsm == 1 else 1)
    for kind in ('U16', 'U32'):
        getattr(sdk, 'registerWrite'+kind).assert_not_called()
    device.close_port()


def test_shutdown_during_native_open_does_not_read_or_publish_after_close(startup_gui, monkeypatch):
    ui = startup_gui
    device = Mock()
    def stop_during_open():
        ui._laser_stopping = True
        ui._close_laser_device()
        return 0
    device.open_port.side_effect = stop_during_open
    monkeypatch.setattr(gui_laser_control, 'NativeLaser', Mock(return_value=device))
    assert not ui._open_existing_laser_native('FAKE')
    assert ui.laser_device is None
    device.StatusRead.assert_not_called()
    device.close_port.assert_called_once()
    ui._publish_laser_state.assert_not_called()
    ui._sync_controls_from_device.assert_not_called()
    ui._set_laser_disconnected_banner.assert_not_called()
    ui._laser_status_timer.start.assert_not_called()
