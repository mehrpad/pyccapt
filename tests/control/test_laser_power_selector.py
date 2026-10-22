"""Manual watt/AOM selection never replays edits or changes native state."""
from unittest.mock import Mock
from types import SimpleNamespace

import pytest
from PyQt6 import QtWidgets

from pyccapt.control.nkt_photonics import nktpbus_switch
from test_laser_manual_contract import laser_gui, mode_gui, watt_gui  # noqa: F401
from test_responsive_gui import app, instrument_window  # noqa: F401


@pytest.mark.parametrize('ceiling,maximum', [(5.255, 5.25), (5.2, 5.2), (.009, 0.)])
def test_two_decimal_watt_editor_does_not_round_a_hardware_limit_upward(app, ceiling, maximum):
    from pyccapt.control.gui.gui_laser_control import Ui_Laser_Control
    ui = Ui_Laser_Control.__new__(Ui_Laser_Control)
    ui.conf = {'laser_ir_max_power_w': ceiling}
    parent = QtWidgets.QWidget()
    ui.gridLayout_3 = QtWidgets.QGridLayout(parent)
    ui.label = QtWidgets.QLabel(parent)
    ui.laser_power = QtWidgets.QDoubleSpinBox(parent)
    ui._setup_laser_power_controls(parent)
    assert ui.laser_ir_power_control.maximum() == maximum
    assert ui.laser_ir_power_control.decimals() == 2
    assert ui.laser_ir_power_control.singleStep() == .1


@pytest.mark.parametrize('wavelength', ['IR', 'Green', 'DUV'])
def test_native_power_modes_are_exclusive_and_switch_without_hardware_writes(watt_gui, monkeypatch, wavelength):
    ui, device, _ = watt_gui
    device.protocol = 'nktpbus'
    device.code = 129
    device.wavelength_basis = 'Verified selection'
    device.wavelength_read = lambda: f'Current position={wavelength}'
    device.set_ir_power_watts = Mock(return_value=1.25)
    ui._laser_power_control_mode = 'watts'
    ui.laser_power_mode_button = QtWidgets.QPushButton()
    ui._use_native_ir_control = Mock(side_effect=AssertionError('Existing native connection must be retained'))
    ui._apply_button_locks_for_status(device.StatusRead())
    assert ui.laser_ir_power_control.isEnabled() and not ui.laser_power.isEnabled()
    assert ui.laser_power_mode_button.text() == 'Use AOM (%)'
    ui.laser_power_changed()
    assert not ui.change_laser_power
    # A late queued AOM callback is rejected independently of widget enabled state.
    ui.change_laser_power = True
    ui._apply_laser_settings(129)
    assert not device.calls
    ui._laser_watt_edit_pending = True
    ui._toggle_laser_power_control()
    assert ui._laser_power_control_mode == 'aom'
    assert ui.laser_power.isEnabled() and not ui.laser_ir_power_control.isEnabled()
    assert not ui._laser_watt_edit_pending
    ui._apply_ir_power_watts()
    device.set_ir_power_watts.assert_not_called()
    ui.change_laser_power = True
    ui._toggle_laser_power_control()
    assert ui._laser_power_control_mode == 'watts'
    assert not ui.change_laser_power
    assert ui.laser_ir_power_control.isEnabled() and not ui.laser_power.isEnabled()
    assert device.code == 129 and not device.calls
    ui._use_native_ir_control.assert_not_called()
    device.close_port.assert_not_called()
    ui.laser_ir_power_control.setValue(1.25)
    ui._mark_laser_watt_edit(1.25)
    ui._commit_laser_watt_edit()
    device.set_ir_power_watts.assert_called_once_with(1.25)


@pytest.mark.parametrize('blocked', ['experiment', 'alignment', 'transition', 'poll', 'stopping'])
def test_power_mode_does_not_change_during_inflight_operations(watt_gui, blocked):
    ui, device, _ = watt_gui
    device.protocol = 'nktpbus'
    ui._laser_power_control_mode = 'watts'
    ui.variables.start_flag = blocked == 'experiment'
    ui.variables.laser_alignment_status = {'active': blocked == 'alignment'}
    ui._laser_mode_switch_in_progress = blocked == 'transition'
    ui._laser_status_in_progress = blocked == 'poll'
    ui._laser_stopping = blocked == 'stopping'
    ui._toggle_laser_power_control()
    assert ui._laser_power_control_mode == 'watts'
    assert not device.calls


@pytest.mark.parametrize('connects', [True, False])
def test_selecting_watts_from_cli_connects_explicitly_without_replaying_edits(watt_gui, connects):
    ui, device, _ = watt_gui
    ui._laser_power_control_mode = 'aom'
    ui.change_laser_power = ui._laser_watt_edit_pending = True
    def connect():
        assert not ui.change_laser_power and not ui._laser_watt_edit_pending
        if connects:
            device.protocol = 'nktpbus'
            device.code = 9
            ui._last_laser_status = device.StatusRead()
        return connects
    ui._use_native_ir_control = Mock(side_effect=connect)
    ui._toggle_laser_power_control()
    ui._use_native_ir_control.assert_called_once()
    assert ui._laser_power_control_mode == ('watts' if connects else 'aom')
    assert ui.laser_ir_power_control.isEnabled() is connects
    assert ui.laser_power.isEnabled() is not connects
    assert not device.calls
    device.close_port.assert_not_called()


@pytest.mark.parametrize('method', ['switch_to_cli_clicked', 'switch_to_nktpbus_mode'])
def test_protocol_handlers_require_override_even_when_called_directly(mode_gui, monkeypatch, method):
    ui = mode_gui
    ui.flag_super_user_stage = False
    for name in ('switch_to_cli', 'switch_to_nktpbus', 'is_cli_responding'):
        monkeypatch.setattr(nktpbus_switch, name, Mock(side_effect=AssertionError('Override is required')))
    getattr(ui, method)()
    assert 'Override Access' in ui.errors[-1]
    ui.laser_device.close_port.assert_not_called()
    assert not ui.laser_device.calls
    ui._update_laser_mode_buttons()
    assert not ui.switch_to_cli_button.isEnabled() and not ui.nktpbus_mode_switch.isEnabled()
    ui.flag_super_user_stage = True
    ui._update_laser_mode_buttons()
    assert ui.switch_to_cli_button.isEnabled() and ui.nktpbus_mode_switch.isEnabled()
    ui.flag_super_user_stage = False
    ui._update_laser_mode_buttons()
    assert not ui.switch_to_cli_button.isEnabled() and not ui.nktpbus_mode_switch.isEnabled()


@pytest.mark.parametrize('instrument_window', ['laser'], indirect=True)
def test_real_selector_defaults_to_watts_and_routes_only_the_active_editor(instrument_window):
    from PyQt6 import QtCore
    from PyQt6.QtTest import QTest
    window, ui, qt_app = instrument_window
    value = SimpleNamespace(watts=1., aom=800)
    watts = Mock(side_effect=lambda requested: setattr(value, 'watts', requested) or requested)
    aom = Mock(side_effect=lambda requested: setattr(value, 'aom', requested))
    ui.laser_device = SimpleNamespace(
        protocol='nktpbus', wavelength_basis='Verified IR',
        StatusRead=lambda: 'ly_oxp2_dev_status 129', ModeRead=lambda: 'e_mode of AOM: 2',
        freq_avaliable=lambda: 'e_freq=4 --> 400000 Hz', FreqRead=lambda: 'Frequency index: 4',
        DivRead=lambda: 'Divider: 1', AOMRead=lambda: f'AOM relative setting: {value.aom}',
        wavelength_read=lambda: 'Current position=IR', read_average_power=lambda: '1 W',
        ir_power_setpoint_watts=lambda: value.watts, set_ir_power_watts=watts, AOM=aom,
    )
    ui._sync_controls_from_device(initial=True)
    window.show()
    qt_app.processEvents()
    assert ui._laser_power_control_mode == 'watts'
    assert ui.laser_ir_power_control.isEnabled() and not ui.laser_power.isEnabled()
    field = ui.laser_ir_power_control
    assert field.decimals() == 2 and field.singleStep() == .1
    assert ui.laser_power.decimals() == 2 and ui.laser_power.singleStep() == .1
    field.setFocus()
    QTest.keyClick(field, QtCore.Qt.Key.Key_A, QtCore.Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(field, '1.25')
    watts.assert_not_called()
    QTest.keyClick(field, QtCore.Qt.Key.Key_Return)
    watts.assert_called_once_with(1.25)
    ui.laser_power_mode_button.click()
    assert ui.laser_power.isEnabled() and not field.isEnabled()
    ui.laser_power.setValue(25.)
    ui.check_laser_status()
    aom.assert_called_once_with(1000)
    ui.laser_power_mode_button.click()
    assert field.isEnabled() and not ui.laser_power.isEnabled()
    assert watts.call_count == 1 and aom.call_count == 1
    assert not ui.switch_to_cli_button.isEnabled() and not ui.nktpbus_mode_switch.isEnabled()
