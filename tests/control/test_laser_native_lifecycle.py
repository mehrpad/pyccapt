"""Native GUI transitions against firmware-like registers; no serial hardware."""
import struct
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt6 import QtWidgets

from pyccapt.control.devices.nkt_native import NativeLaser
from pyccapt.control.gui import laser_power_control
from pyccapt.control.nkt_photonics import nktpbus_switch, readback

from test_laser_manual_contract import laser_gui, mode_gui, watt_gui  # noqa: F401


@pytest.fixture
def lifecycle_native(watt_gui, monkeypatch, request):
    ui, cli, conversion = watt_gui
    wavelength = getattr(request, 'param', 0)
    registers = {0x01: 1, 0x67: 0, 0x32: 1, 0x37: 0, 0x20: 1000,
                 0x05: 800, 0x35: 4, 0x36: 10, 0x10: 250, 0x11: 50, 0x0E: wavelength}
    writes = []
    sdk = Mock()
    sdk.openPorts.return_value = sdk.closePorts.return_value = 0
    sdk.deviceGetAllTypes.return_value = (0, bytes(15) + b'\x95' + bytes(239))
    sdk.registerRead.return_value = (0, struct.pack('<13I', *([0]*4 + [400000] + [0]*8)))
    def read(port, address, register, index):
        return 0, registers[register]
    def write(kind, port, address, register, value, index):
        writes.append((kind, register, value))
        if register == 0x30 and value == 1:
            registers.update({0x01: 1, 0x32: 1})
        elif register == 0x30 and value == 3:
            assert registers[0x01] == 1 and registers[0x32] in (1, 2)
            registers[0x01] = 2  # Firmware accepts Standby, then warms up.
        elif register == 0x30 and value == 6:
            assert registers[0x01] == 3 and registers[0x32] == 2
            registers[0x01] = 6
        elif register == 0x32:
            assert (kind, value, registers[0x01], registers[0x32]) == ('U8', 1, 1, 1)
            # Accept the request, but leave relay confirmation asynchronous.
        elif register == 0x20:
            assert kind == 'U16'
            registers[register] = value
        elif register == 0x0E:
            assert kind == 'U8' and registers[0x01] == 1
            registers[register] = value
        else:
            pytest.fail(f'Unexpected lifecycle write: {kind}, 0x{register:02X}={value}')
        return 0
    for kind in ('U8', 'U16', 'U32'):
        getattr(sdk, 'registerRead'+kind).side_effect = read
        getattr(sdk, 'registerWrite'+kind).side_effect = (
            lambda *args, kind=kind: write(kind, *args))
    monkeypatch.setattr(nktpbus_switch, '_load_nktp_dll_module', lambda: sdk)
    for method in ('switch_to_cli', 'switch_to_nktpbus', 'request_cli_protocol'):
        monkeypatch.setattr(nktpbus_switch, method, Mock(side_effect=AssertionError('No lifecycle protocol changes')))
    ui._close_laser_device()
    native = NativeLaser('FAKE', wavelength=wavelength)
    assert native.open_port() == 0
    assert not writes
    ui.laser_device = native
    ui.laser_native_control_button = QtWidgets.QPushButton()
    ui._publish_laser_state(native.StatusRead())
    assert ui._sync_controls_from_device(initial=True)
    yield SimpleNamespace(ui=ui, native=native, sdk=sdk, registers=registers,
                          writes=writes, cli=cli, conversion=conversion)
    native.close_port()


@pytest.mark.parametrize('lifecycle_native', [0, 1, 3], indirect=True, ids=['IR', 'Green', 'DUV'])
def test_native_listen_standby_warmup_emission_and_live_watts(lifecycle_native):
    case = lifecycle_native
    ui, native, registers = case.ui, case.native, case.registers
    assert native.standby_available
    assert native.status_fault == 'interlock waiting for reset'
    assert ui.laser_standby.isEnabled()
    assert ui.led_laser_listen.pixmap().cacheKey() == ui.led_green.cacheKey()

    ui.laser_standby_clicked()
    ui.check_laser_status()
    assert case.writes == [('U8', 0x30, 3)]
    assert registers[0x01] == 2 and registers[0x32] == 1
    assert readback.scalar(ui._last_laser_status) == 17
    assert ui._laser_standby_pending
    assert ui.led_laser_listen.pixmap().cacheKey() == ui.led_green.cacheKey()
    before = ui.led_laser_laser_standby.pixmap().cacheKey()
    ui._blink_laser_indicators()
    assert {before, ui.led_laser_laser_standby.pixmap().cacheKey()} == {
        ui.led_red.cacheKey(), ui.led_orange.cacheKey()}
    assert not ui.laser_on.isEnabled()

    # Warm-up completion is reported by firmware, not assumed from the ACK.
    registers.update({0x01: 3, 0x32: 2})
    ui.check_laser_status()
    assert case.writes == [('U8', 0x30, 3)]
    assert not ui._laser_standby_pending
    assert ui.led_laser_laser_standby.pixmap().cacheKey() == ui.led_green.cacheKey()
    assert ui.laser_on.isEnabled()

    ui.laser_on_clicked()
    ui.check_laser_status()
    assert registers[0x01] == 6 and ui.laser_ir_power_control.isEnabled()
    for watts in (1.25, 1.5):
        ui.laser_ir_power_control.setValue(watts)
        ui._mark_laser_watt_edit(watts)
        ui._commit_laser_watt_edit()
        assert registers[0x01] == 6
        assert ui.laser_ir_power_control.value() == watts
    assert case.writes == [('U8', 0x30, 3), ('U8', 0x30, 6),
                           ('U16', 0x20, 1250), ('U16', 0x20, 1500)]
    case.sdk.openPorts.assert_called_once_with('FAKE', 1, 1)
    case.sdk.closePorts.assert_not_called()
    case.cli.close_port.assert_called_once()
    case.conversion.assert_not_called()
    ui._open_laser_cli.assert_not_called()
    assert ui.laser_device is native


def test_return_to_listen_enables_native_wavelength_then_preserves_live_watts(lifecycle_native):
    case = lifecycle_native
    ui, registers = case.ui, case.registers
    ui._laser_power_control_mode = 'watts'
    registers.update({0x01: 6, 0x32: 2})
    ui._publish_laser_state(case.native.StatusRead())
    ui._apply_button_locks_for_status(case.native.StatusRead())
    assert not ui.laser_wavelegnth.isEnabled()
    ui.laser_listen_clicked()
    ui.check_laser_status()
    assert ui.laser_wavelegnth.isEnabled() and ui.laser_standby.isEnabled()
    for selection in ('Green', 'DUV', 'IR'):
        ui.laser_wavelegnth.setCurrentText(selection)
        ui.laser_wavelegnth_changed()
        ui.check_laser_status()
        assert ui.laser_wavelegnth.currentText() == selection
        assert ui.laser_wavelegnth.isEnabled()
        assert registers[0x01] == 1 and registers[0x32] == 1
    assert case.writes == [('U8', 0x30, 1), ('U8', 0x0E, 1), ('U8', 0x0E, 3), ('U8', 0x0E, 0)]
    ui.laser_standby_clicked()
    ui.check_laser_status()
    registers.update({0x01: 3, 0x32: 2})
    ui.check_laser_status()
    ui.laser_on_clicked()
    ui.check_laser_status()
    ui.laser_ir_power_control.setValue(1.5)
    ui._mark_laser_watt_edit(1.5)
    ui._commit_laser_watt_edit()
    assert registers[0x01] == 6 and registers[0x20] == 1500
    case.sdk.openPorts.assert_called_once()
    case.sdk.closePorts.assert_not_called()


@pytest.mark.parametrize('fails', [False, True])
def test_watt_transaction_blocks_nested_timer_poll_and_releases_guard(watt_gui, fails):
    ui, device, _ = watt_gui
    device.protocol = 'nktpbus'
    device.code = 129
    device.wavelength_basis = 'IR verified'
    ui._laser_power_control_mode = 'watts'
    ui._apply_button_locks_for_status(device.StatusRead())
    assert ui.laser_on.isEnabled() and ui.laser_enable.isEnabled()
    ui.check_laser_status = Mock(side_effect=AssertionError('No nested SDK poll during watt write'))
    def write(watts):
        assert ui._laser_status_in_progress
        ui._poll_laser_status()  # SDK/Windows can process this timer while writing.
        assert ui.laser_on.isEnabled() and ui.laser_enable.isEnabled()
        if fails:
            raise ConnectionError('Write failed')
        return watts
    device.set_ir_power_watts = Mock(side_effect=write)
    ui.laser_ir_power_control.setValue(1.25)
    ui._apply_ir_power_watts()
    ui.check_laser_status.assert_not_called()
    assert not ui._laser_status_in_progress
    assert ui.laser_on.isEnabled() and ui.laser_enable.isEnabled()
    assert not device.calls and device.code == 129
    if fails:
        assert 'Write failed' in ui.errors[-1]


@pytest.mark.parametrize('confirmed', [True, False], ids=['delayed-confirmation', 'timeout'])
def test_explicit_reset_waits_for_later_readback_without_automatic_standby(lifecycle_native, monkeypatch, confirmed):
    case = lifecycle_native
    ui = case.ui
    clock = [100.]
    monkeypatch.setattr(laser_power_control, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    assert ui._reset_native_interlock()
    assert case.writes == [('U8', 0x32, 1)]
    assert ui._laser_interlock_reset_deadline == 105.
    assert not ui.standby_mode and not ui.on_mode
    assert ui.laser_native_control_button.text() == 'Resetting interlock...'

    clock[0] += 1.2  # Longer than the former half-second synchronous wait.
    ui.check_laser_status()
    assert ui._laser_interlock_reset_deadline == 105.
    assert case.writes == [('U8', 0x32, 1)]
    if confirmed:
        case.registers[0x32] = 2
        clock[0] += 1.2
    else:
        clock[0] = 106.
    ui.check_laser_status()
    assert ui._laser_interlock_reset_deadline is None
    assert case.writes == [('U8', 0x32, 1)]
    assert case.registers[0x01] == 1
    assert not ui.standby_mode and not ui.on_mode
    assert ('Interlock reset confirmed' if confirmed else 'reset was not confirmed') in ui.errors[-1]
    case.sdk.openPorts.assert_called_once_with('FAKE', 1, 1)
    case.sdk.closePorts.assert_not_called()
