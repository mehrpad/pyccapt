"""Real native adapter with simulated independent IR/SHG/FHG registers."""
from unittest.mock import Mock

import pytest

from pyccapt.control.gui import gui_laser_control
from pyccapt.control.nkt_photonics import readback
from test_laser_native_control import native  # noqa: F401
from test_laser_manual_contract import laser_gui, mode_gui, watt_gui  # noqa: F401


def attach_monitors(laser, sdk, registers, modules):
    types = bytearray(256)
    types[laser.address] = 0x95
    for address, (kind, values) in modules.items():
        types[address] = kind
    sdk.deviceGetAllTypes.return_value = (0, bytes(types))
    read = sdk.registerReadU8.side_effect
    def query(port, address, register, index):
        if address == laser.address:
            return read(port, address, register, index)
        return 0, modules[address][1][register]
    sdk.registerReadU8.side_effect = sdk.registerReadU16.side_effect = query


@pytest.mark.parametrize('wavelength,watts', [(1, .234), (3, .017), (3, 0.)])
def test_selected_monitor_uses_harmonic_module_and_correct_milliwatt_units(native, wavelength, watts):
    laser, sdk, registers = native
    registers[0x0E] = wavelength
    modules = {4: (0x93, {0x32: 2, 0x10: 234}),
               5: (0x93, {0x32: 4, 0x10: round(watts*1000) if wavelength == 3 else 17})}
    attach_monitors(laser, sdk, registers, modules)
    reply = laser.power_read_dv_green()
    assert readback.quantity(reply) == (watts, 'W')
    assert readback.selected_harmonic_reply(reply, wavelength) == reply
    assert '0.001 W/count' in laser.harmonic_power_basis
    assert f'module 0x{4 if wavelength == 1 else 5:02X}' in laser.harmonic_power_basis
    sdk.registerWriteU8.assert_not_called()
    sdk.registerWriteU16.assert_not_called()


@pytest.mark.parametrize('kind,wavelength,values,expected', [
    (0x86, 1, {0x10: 37}, .37),
    (0x90, 1, {0x10: 370}, .37),
    (0x8D, 1, {0x30: 1, 0x10: 370, 0x11: 17}, .37),
    (0x8D, 3, {0x30: 3, 0x10: 370, 0x11: 17}, .017),
])
def test_other_vendor_harmonic_models_have_model_specific_units(native, kind, wavelength, values, expected):
    laser, sdk, registers = native
    registers[0x0E] = wavelength
    attach_monitors(laser, sdk, registers, {4: (kind, values)})
    assert readback.quantity(laser.power_read_dv_green()) == (expected, 'W')


@pytest.mark.parametrize('ambiguous', [False, True])
def test_missing_or_duplicate_selected_monitor_is_unknown_instead_of_using_ir(native, ambiguous):
    laser, sdk, registers = native
    registers[0x0E] = 3
    modules = {4: (0x93, {0x32: 2, 0x10: 234})}
    if ambiguous:
        modules = {4: (0x93, {0x32: 4, 0x10: 17}), 5: (0x93, {0x32: 4, 0x10: 99})}
    attach_monitors(laser, sdk, registers, modules)
    with pytest.raises(ValueError, match='one matching native harmonic monitor'):
        laser.power_read_dv_green()
    assert not laser.harmonic_power_basis
    sdk.registerWriteU8.assert_not_called()


@pytest.mark.parametrize('wavelength,name,watts', [(1, 'Green', .234), (3, 'DUV', .017)])
def test_gui_displays_selected_output_watts_and_calculated_pulse_energy(native, watt_gui, wavelength, name, watts):
    laser, sdk, registers = native
    ui, _, _ = watt_gui
    registers.update({0x0E: wavelength, 0x36: 2, 0x10: 594})
    attach_monitors(laser, sdk, registers, {
        4: (0x93, {0x32: 2, 0x10: 234}), 5: (0x93, {0x32: 4, 0x10: 17})})
    ui.laser_device = laser
    assert ui._sync_controls_from_device(initial=True)
    assert ui.laser_wavelegnth.currentText() == name
    assert ui.laser_ir_power_disp.value() == pytest.approx(.594)
    assert ui.laser_power_disp.value() == pytest.approx(watts)
    assert ui.laser_pulse_energy_disp.value() == pytest.approx(watts / 200000. * 1e6)
    assert ui.variables.laser_telemetry['valid']
    assert name in ui.variables.laser_telemetry['source']
    assert 'pulse energy calculated' in ui.laser_pulse_energy_disp.toolTip()
    assert registers[0x01] == 6
    sdk.registerWriteU8.assert_not_called()
    sdk.registerWriteU16.assert_not_called()


@pytest.mark.parametrize('interlock,error', [(1, 0), (2, 0), (0x3000, 0), (1, 7)])
def test_zero_in_listen_clears_real_stored_settings_without_reset_or_emission(native, interlock, error):
    laser, sdk, registers = native
    registers.update({0x01: 1, 0x32: interlock, 0x67: error, 0x20: 3000, 0x05: 2400})
    laser.zero_power_in_listen()
    assert registers[0x20] == registers[0x05] == 0
    assert [call.args[2:4] for call in sdk.registerWriteU16.call_args_list] == [(0x20, 0), (0x05, 0)]
    sdk.registerWriteU8.assert_not_called()
    assert registers[0x32] == interlock and registers[0x01] == 1


@pytest.mark.parametrize('state', [2, 3, 5, 6])
def test_listen_zero_waits_for_confirmed_state(native, state):
    laser, sdk, registers = native
    registers[0x01] = state
    with pytest.raises(ValueError, match='Listen is confirmed'):
        laser.zero_power_in_listen()
    sdk.registerWriteU16.assert_not_called()


def test_delayed_listen_clears_once_after_confirmation_and_discards_old_edit(native, watt_gui):
    laser, sdk, registers = native
    ui, _, _ = watt_gui
    registers[0x20] = 3000
    ui.laser_device = laser
    assert ui._sync_controls_from_device(initial=True)
    ui._laser_watt_edit_pending = ui.change_laser_power = True
    ui.laser_listen_clicked()
    assert not ui._laser_watt_edit_pending and not ui.change_laser_power
    ui.check_laser_status()  # Listen command ACK, but simulated FSM is still emitting.
    assert ui.laser_ir_power_control.value() == 3.
    sdk.registerWriteU16.assert_not_called()
    registers.update({0x01: 1, 0x32: 1})
    ui.check_laser_status()
    assert ui.laser_ir_power_control.value() == 0. and ui.laser_power.value() == 0.
    assert registers[0x20] == registers[0x05] == 0
    assert not ui._laser_power_zero_pending
    ui.check_laser_status()
    assert sdk.registerWriteU16.call_count == 2
    sdk.registerWriteU8.assert_called_once_with('FAKE', 15, 0x30, 1, -1)


def test_startup_already_in_native_listen_clears_old_watts_without_protocol_switch(native, watt_gui, monkeypatch):
    laser, sdk, registers = native
    ui, _, _ = watt_gui
    registers.update({0x01: 1, 0x32: 1, 0x20: 3000})
    ui._close_laser_device()
    monkeypatch.setattr(gui_laser_control, 'NativeLaser', lambda *args, **kwargs: laser)
    assert ui._open_existing_laser_native('FAKE')
    assert ui.laser_device is laser and ui.laser_ir_power_control.value() == 0.
    assert registers[0x20] == registers[0x05] == 0
    assert ui.laser_standby.isEnabled()
    sdk.registerWriteU8.assert_not_called()
    sdk.openPorts.assert_called_once()


def test_failed_zero_is_reported_instead_of_faking_zero_in_the_editor(native, watt_gui):
    laser, sdk, registers = native
    ui, _, _ = watt_gui
    registers[0x20] = 3000
    ui.laser_device = laser
    assert ui._sync_controls_from_device(initial=True)
    registers.update({0x01: 1, 0x32: 1})
    sdk.registerWriteU16.side_effect = None
    sdk.registerWriteU16.return_value = 0  # Simulate refused zero settings.
    ui.laser_listen_clicked()
    ui._poll_laser_status()
    assert ui.laser_ir_power_control.value() == 3.
    assert any('Zero watt/AOM settings were not confirmed' in message for message in ui.errors)
    assert not ui.laser_ir_power_control.isEnabled()


def test_shutdown_during_listen_zero_does_not_publish_to_closed_manager(native, watt_gui):
    from test_laser_readout_shutdown import ClosedVariables
    laser, _, registers = native
    ui, _, _ = watt_gui
    ui.laser_device = laser
    ui._laser_power_zero_pending = True
    registers.update({0x01: 1, 0x32: 1})
    status = laser.StatusRead()
    closed = ClosedVariables()
    def shutdown():
        ui._laser_stopping = True
        ui.variables = closed
    laser.zero_power_in_listen = Mock(side_effect=shutdown)
    ui._publish_laser_state(status)
    assert closed.accesses == []
