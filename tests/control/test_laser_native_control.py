"""Persistent native control, using simulated NKT SDK register responses."""
import struct
from unittest.mock import Mock

import pytest

from pyccapt.control.devices.nkt_native import NativeLaser
from pyccapt.control.nkt_photonics import nktpbus_switch, readback


@pytest.fixture
def native(monkeypatch, request):
    wavelength = getattr(request, 'param', 0)
    registers = {0x01: 6, 0x67: 0, 0x32: 2, 0x37: 0, 0x20: 1000,
                 0x05: 800, 0x35: 4, 0x36: 1, 0x10: 1000, 0x11: 250}
    sdk = Mock()
    sdk.openPorts.return_value = sdk.closePorts.return_value = 0
    sdk.deviceGetAllTypes.return_value = (0, bytes(15) + b'\x95' + bytes(239))
    sdk.registerRead.return_value = (0, struct.pack('<13I', *([0]*4 + [400000] + [0]*8)))
    def read(port, address, register, index):
        if register == 0x0E and register not in registers:
            return 4, 0
        return 0, registers[register]
    def write(port, address, register, value, index):
        registers[register] = value
        if register == 0x20:
            registers[0x05] = value // 2  # Simulated firmware calibration.
        return 0
    for kind in ('U8', 'U16', 'U32'):
        getattr(sdk, 'registerRead'+kind).side_effect = read
        getattr(sdk, 'registerWrite'+kind).side_effect = write
    monkeypatch.setattr(nktpbus_switch, '_load_nktp_dll_module', lambda: sdk)
    laser = NativeLaser('FAKE', wavelength=wavelength)
    assert laser.open_port() == 0
    assert readback.wavelength(laser.wavelength_read()) == wavelength
    yield laser, sdk, registers
    laser.close_port()


@pytest.mark.parametrize('state', [1, 3, 5, 6])
@pytest.mark.parametrize('native', [0, 1, 3], indirect=True)
def test_watts_preserve_state_shutter_and_sdk_session(native, state):
    laser, sdk, registers = native
    selection = laser.wavelength_read()
    registers[0x01] = state
    assert laser.set_ir_power_watts(1.2) == 1.2
    assert laser.set_ir_power_watts(1.4) == 1.4
    assert registers[0x01] == state
    assert registers[0x05] == 700
    assert laser.wavelength_read() == selection
    assert sdk.openPorts.call_count == 1
    sdk.closePorts.assert_not_called()
    sdk.registerWriteU8.assert_not_called()  # No FSM, shutter, mode or protocol write.
    sdk.registerWriteU32.assert_not_called()
    assert [call.args[2:4] for call in sdk.registerWriteU16.call_args_list] == [(0x20, 1200), (0x20, 1400)]


@pytest.mark.parametrize('native', [None], indirect=True)
def test_native_attach_without_cli_wavelength_does_not_invent_selected_output(native):
    laser, sdk, registers = native
    assert readback.wavelength(laser.wavelength_read()) is None
    assert 'not been verified' in laser.wavelength_basis
    assert readback.scalar(laser.StatusRead()) == 129
    sdk.registerWriteU8.assert_not_called()
    sdk.registerWriteU16.assert_not_called()
    assert laser.set_ir_power_watts(1.4) == 1.4
    assert registers[0x01] == 6
    sdk.registerWriteU8.assert_not_called()


@pytest.mark.parametrize('native', [None], indirect=True)
@pytest.mark.parametrize('wavelength', [0, 1, 3])
def test_existing_native_wavelength_is_read_without_a_protocol_change(native, wavelength):
    laser, sdk, registers = native
    registers[0x0E] = wavelength
    assert readback.wavelength(laser.wavelength_read()) == wavelength
    assert 'register 0x0E' in laser.wavelength_basis
    sdk.registerWriteU8.assert_not_called()
    sdk.registerWriteU16.assert_not_called()


def test_persistent_native_session_keeps_vendor_watchdog_monitoring_active(native):
    _, sdk, _ = native
    sdk.openPorts.assert_called_once_with('FAKE', 1, 1)


def test_explicit_interlock_reset_only_writes_vendor_reset_and_waits_for_later_readback(native):
    laser, sdk, registers = native
    registers.update({0x01: 1, 0x32: 1})
    laser.StatusRead()
    assert laser.interlock_reset_available
    laser.reset_interlock()
    sdk.registerWriteU8.assert_called_once_with('FAKE', 15, 0x32, 1, -1)
    sdk.registerWriteU16.assert_not_called()
    assert readback.scalar(laser.StatusRead()) == 9
    assert registers[0x32] == 1  # An ACK is not relay confirmation.
    registers[0x32] = 2
    assert readback.scalar(laser.StatusRead()) == 9
    assert not laser.status_fault and not laser.interlock_reset_available
    assert registers[0x01] == 1


@pytest.mark.parametrize('state,interlock,error', [
    (1, 0, 0), (1, 2, 0), (1, 0x0300, 0), (1, 0x3000, 0), (1, 1, 7),
    (2, 1, 0), (3, 1, 0), (5, 1, 0), (6, 1, 0),
])
def test_reset_rechecks_state_and_never_overrides_open_or_failed_interlock(native, state, interlock, error):
    laser, sdk, registers = native
    registers.update({0x01: 1, 0x32: 1})
    laser.StatusRead()
    assert laser.interlock_reset_available
    registers.update({0x01: state, 0x32: interlock, 0x67: error})
    with pytest.raises(ValueError):
        laser.reset_interlock()
    sdk.registerWriteU8.assert_not_called()
    sdk.registerWriteU16.assert_not_called()


@pytest.mark.parametrize('watts', [-1., 5.201, float('inf'), float('nan')])
def test_invalid_watts_never_write(native, watts):
    laser, sdk, _ = native
    with pytest.raises(ValueError):
        laser.set_ir_power_watts(watts)
    sdk.registerWriteU16.assert_not_called()


@pytest.mark.parametrize('register,value', [(0x37, 1), (0x67, 7), (0x32, 1), (0x01, 2)])
def test_external_control_fault_interlock_and_warmup_block_watts(native, register, value):
    laser, sdk, registers = native
    registers[register] = value
    with pytest.raises(ValueError):
        laser.set_ir_power_watts(1.)
    sdk.registerWriteU16.assert_not_called()
    sdk.registerWriteU8.assert_not_called()


def test_restricted_relative_cap_is_not_exceeded_via_watt_conversion(native):
    laser, sdk, _ = native
    laser.max_aom_percent = 20
    with pytest.raises(ValueError, match='relative AOM cap'):
        laser.set_ir_power_watts(1.)
    sdk.registerWriteU16.assert_not_called()
    assert laser.set_ir_power_watts(0.) == 0.


def test_aom_control_and_documented_units_are_retained(native):
    laser, sdk, registers = native
    laser.AOM(1200)
    assert readback.scalar(laser.AOMRead()) == 1200
    assert readback.quantity(laser.read_average_power()) == (1., 'W')
    assert readback.quantity(laser.read_pulse_energy()) == (2500., 'nJ')
    assert laser.ir_power_setpoint_watts() == 1.
    assert readback.frequency_table(laser.freq_avaliable()) == {4: 400000.}
    sdk.registerWriteU16.assert_called_once_with('FAKE', 15, 0x05, 1200, -1)
    sdk.registerWriteU8.assert_not_called()


@pytest.mark.parametrize('state,code', [(0,1),(1,9),(2,17),(3,33),(4,17),(5,65),(6,129)])
def test_native_fsm_uses_existing_gui_state_contract(native, state, code):
    laser, _, registers = native
    registers[0x01] = state
    assert readback.scalar(laser.StatusRead()) == code


def test_setter_failure_is_not_success_and_does_not_switch_protocol(native):
    laser, sdk, _ = native
    sdk.registerWriteU16.side_effect = None
    sdk.registerWriteU16.return_value = 6
    with pytest.raises(ConnectionError):
        laser.set_ir_power_watts(1.)
    sdk.registerWriteU8.assert_not_called()
    sdk.closePorts.assert_not_called()


def test_port_is_closed_on_failed_native_discovery(monkeypatch):
    sdk = Mock()
    sdk.openPorts.return_value = sdk.closePorts.return_value = 0
    sdk.deviceGetAllTypes.return_value = (0, bytes(255))
    monkeypatch.setattr(nktpbus_switch, '_load_nktp_dll_module', lambda: sdk)
    laser = NativeLaser('FAKE', wavelength=0)
    assert laser.open_port() == -1
    sdk.closePorts.assert_called_once_with('FAKE')
    assert not laser.connected


def test_failed_close_retains_sdk_ownership_and_prevents_double_open(native):
    laser, sdk, _ = native
    sdk.closePorts.return_value = 6
    with pytest.raises(ConnectionError):
        laser.close_port()
    assert laser._opened and not laser.connected
    assert laser.open_port() == -1
    assert sdk.openPorts.call_count == 1
    sdk.closePorts.return_value = 0
    laser.close_port()
    assert not laser._opened


def test_unaccepted_setpoint_is_reported_without_state_commands(native):
    laser, sdk, _ = native
    sdk.registerWriteU16.side_effect = None
    sdk.registerWriteU16.return_value = 0  # Simulate firmware ignoring the write.
    with pytest.raises(ValueError, match='not accepted'):
        laser.set_ir_power_watts(1.4)
    sdk.registerWriteU8.assert_not_called()
    sdk.closePorts.assert_not_called()


@pytest.mark.parametrize('interlock,error', [(1, 0), (0x3000, 0), (2, 7)])
def test_listen_is_observable_during_fault_but_cannot_raise_power(native, interlock, error):
    laser, sdk, registers = native
    registers.update({0x01: 1, 0x32: interlock, 0x67: error})
    assert readback.scalar(laser.StatusRead()) == 9
    assert laser.status_fault
    assert laser.status_details == dict(fsm=1, interlock=interlock, error=error)
    actions = [laser.Enable, laser.AOMEnable, lambda: laser.AOM(800), lambda: laser.set_ir_power_watts(1.)]
    if interlock != 1 or error:
        actions.append(laser.Standby)
    for action in actions:
        with pytest.raises(ValueError):
            action()
    sdk.registerWriteU8.assert_not_called()
    sdk.registerWriteU16.assert_not_called()
    laser.Listen()
    sdk.registerWriteU8.assert_called_once_with('FAKE', 15, 0x30, 1, -1)
    registers.update({0x32: 2, 0x67: 0})
    assert readback.scalar(laser.StatusRead()) == 9 and not laser.status_fault
    assert laser.set_ir_power_watts(1.) == 1.


@pytest.mark.parametrize('interlock,error', [(0, 0), (0x0300, 0), (0x3000, 0), (0x4000, 0), (0xFF00, 0), (1, 7)])
def test_standby_never_resets_open_or_failed_interlock(native, interlock, error):
    laser, sdk, registers = native
    registers.update({0x01: 1, 0x32: interlock, 0x67: error})
    with pytest.raises(ValueError):
        laser.Standby()
    sdk.registerWriteU16.assert_not_called()
    sdk.registerWriteU8.assert_not_called()


def test_standby_matches_vendor_listen_to_warmup_sequence_without_a_reset(native):
    laser, sdk, registers = native
    registers.update({0x01: 1, 0x32: 1})
    laser.Standby()
    sdk.registerWriteU8.assert_called_once_with('FAKE', 15, 0x30, 3, -1)
    sdk.registerWriteU16.assert_not_called()
    registers[0x01] = 2
    assert readback.scalar(laser.StatusRead()) == 17  # Closed-loop warm-up remains warming.
    registers.update({0x01: 3, 0x32: 2})
    assert readback.scalar(laser.StatusRead()) == 33


@pytest.mark.parametrize('close_result', [0, 2])
@pytest.mark.parametrize('failure', ['read', 'write'])
def test_only_explicit_listen_reopens_a_lost_sdk_port(native, close_result, failure):
    laser, sdk, registers = native
    registers[0x01] = 1
    read = sdk.registerReadU8.side_effect
    write = sdk.registerWriteU8.side_effect
    if failure == 'write':
        sdk.registerWriteU8.side_effect = lambda *args: 13
        with pytest.raises(ConnectionError):
            laser.Listen()
        sdk.registerWriteU8.side_effect = write
        sdk.registerWriteU8.reset_mock()
    sdk.registerReadU8.side_effect = lambda *args: (13, 0)
    with pytest.raises(ConnectionError):
        laser.StatusRead()
    assert not laser.connected and not laser.status_details
    with pytest.raises(ConnectionError):
        laser.set_ir_power_watts(1.)
    assert sdk.openPorts.call_count == 1
    sdk.registerWriteU16.assert_not_called()
    sdk.registerReadU8.side_effect = read
    sdk.closePorts.return_value = close_result
    laser.Listen()
    assert sdk.openPorts.call_count == 2
    sdk.closePorts.assert_called_once_with('FAKE')
    sdk.registerWriteU8.assert_called_once_with('FAKE', 15, 0x30, 1, -1)
    assert laser.connected
