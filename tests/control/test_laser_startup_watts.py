"""Default native watt startup stays in Listen and yields to manual actions."""
from unittest.mock import Mock

import pytest

from pyccapt.control.gui import gui_laser_control, laser_power_control
from pyccapt.control.nkt_photonics import nktpbus_switch
from test_laser_manual_contract import Laser, laser_gui, mode_gui, watt_gui  # noqa: F401


@pytest.fixture
def startup_watts(watt_gui):
    ui, cli, _ = watt_gui
    cli.code = 9
    ui._laser_power_control_mode = 'watts'
    ui._laser_startup_watts_pending = True
    ui._laser_startup_watts_deadline = None
    ui._laser_startup_retry_timer = Mock()
    ui.flag_super_user_stage = False
    ui._reset_native_interlock = Mock(side_effect=AssertionError('No startup interlock reset'))
    return ui, cli


@pytest.mark.parametrize('wavelength,index', [('IR', 0), ('Green', 1), ('DUV', 3)])
@pytest.mark.parametrize('waiting_reset', [False, True])
def test_startup_handover_once_in_listen_without_power_or_emission(
        startup_watts, monkeypatch, wavelength, index, waiting_reset):
    ui, cli = startup_watts
    cli.wavelength_read = lambda: f'Current position={wavelength}'
    native = Laser()
    native.protocol = 'nktpbus'
    native.code = 9
    native.wavelength_basis = 'CLI verified'
    native.wavelength_read = cli.wavelength_read
    native.open_port = Mock(return_value=0)
    native.interlock_reset_available = waiting_reset
    native.status_fault = 'interlock waiting for reset' if waiting_reset else ''
    native.status_details = dict(fsm=1, error=0, interlock=1 if waiting_reset else 2)
    native.standby_available = True
    factory = Mock(return_value=native)
    monkeypatch.setattr(laser_power_control, 'NativeLaser', factory)
    monkeypatch.setattr(nktpbus_switch, '_load_nktp_dll_module', Mock())
    def switch(port):
        cli.close_port.assert_called_once()
        assert ui.laser_device is None
        assert not ui._laser_status_timer.isActive()
    bus = Mock(side_effect=switch)
    monkeypatch.setattr(nktpbus_switch, 'switch_to_nktpbus', bus)

    ui._complete_laser_startup()
    ui._complete_laser_startup()  # Even a queued callback cannot repeat handover/reset.

    bus.assert_called_once_with('FAKE')
    assert factory.call_args.kwargs['wavelength'] == index
    assert ui.laser_device is native and native.code == 9
    assert native.calls == [('Listen',)] and not cli.calls
    native.close_port.assert_not_called()
    ui._reset_native_interlock.assert_not_called()
    assert ui._laser_power_control_mode == 'watts'
    assert not ui.laser_power.isEnabled()
    assert ui.laser_ir_power_control.isEnabled() is not waiting_reset
    assert ui.laser_standby.isEnabled()
    assert not ui.switch_to_cli_button.isEnabled() and not ui.nktpbus_mode_switch.isEnabled()
    assert not ui._laser_startup_watts_pending
    assert ui._laser_status_timer.isActive()


def test_existing_native_connection_does_not_reset_or_reopen(startup_watts):
    ui, device = startup_watts
    device.protocol = 'nktpbus'
    device.interlock_reset_available = True
    ui._use_native_ir_control = Mock(side_effect=AssertionError('Keep existing native connection'))
    ui._complete_laser_startup()
    assert not device.calls and not ui._laser_startup_watts_pending
    device.close_port.assert_not_called()
    ui._use_native_ir_control.assert_not_called()
    ui._reset_native_interlock.assert_not_called()


@pytest.mark.parametrize('status', [17, 33, 65, 129])
def test_startup_waits_for_actual_listen_without_replaying_command(startup_watts, status):
    ui, device = startup_watts
    device.code = status
    ui._use_native_ir_control = Mock()
    ui._complete_laser_startup()
    ui._use_native_ir_control.assert_not_called()
    ui._laser_startup_retry_timer.start.assert_called_once_with(500)
    assert ui._laser_startup_watts_pending and not device.calls
    device.code = 9
    ui._complete_laser_startup()
    ui._use_native_ir_control.assert_called_once_with(listen_only=True)
    assert not ui._laser_startup_watts_pending


def test_missing_listen_confirmation_times_out_without_handover(startup_watts, monkeypatch):
    ui, device = startup_watts
    device.code = 17
    ui._laser_startup_watts_deadline = 99.
    monkeypatch.setattr(gui_laser_control.time, 'monotonic', lambda: 100.)
    ui._use_native_ir_control = Mock()
    ui._complete_laser_startup()
    ui._complete_laser_startup()
    ui._use_native_ir_control.assert_not_called()
    assert not ui._laser_startup_watts_pending and not device.calls
    assert len(ui.errors) == 1 and 'Listen was not confirmed' in ui.errors[0]


def test_lost_cli_status_cancels_startup_without_repeated_warnings(startup_watts):
    ui, device = startup_watts
    device.StatusRead = Mock(side_effect=TimeoutError('No status'))
    ui._use_native_ir_control = Mock()
    ui._complete_laser_startup()
    ui._complete_laser_startup()
    ui._use_native_ir_control.assert_not_called()
    assert len(ui.errors) == 1 and 'No status' in ui.errors[0]
    assert not ui._laser_startup_watts_pending and not device.calls


def test_startup_recovery_connects_before_preparing_watts(startup_watts):
    ui, device = startup_watts
    ui.laser_device = None
    ui._recover_laser_startup = Mock(side_effect=lambda: setattr(ui, 'laser_device', device))
    ui._use_native_ir_control = Mock()
    ui._complete_laser_startup()
    ui._recover_laser_startup.assert_called_once()
    ui._use_native_ir_control.assert_called_once_with(listen_only=True)


@pytest.mark.parametrize('blocked', ['stopping', 'transition', 'experiment', 'alignment', 'aom', 'standby', 'setting'])
def test_startup_yields_to_operations_or_manual_selection(startup_watts, blocked):
    ui, device = startup_watts
    ui._laser_stopping = blocked == 'stopping'
    ui._laser_mode_switch_in_progress = blocked == 'transition'
    ui.variables.start_flag = blocked == 'experiment'
    ui.variables.laser_alignment_status = {'active': blocked == 'alignment'}
    ui._laser_power_control_mode = 'aom' if blocked == 'aom' else 'watts'
    ui.standby_mode = blocked == 'standby'
    ui.change_laser_rate = blocked == 'setting'
    ui._use_native_ir_control = Mock()
    ui._complete_laser_startup()
    assert not ui._laser_startup_watts_pending and not device.calls
    ui._use_native_ir_control.assert_not_called()


def test_startup_defers_while_poll_owns_port(startup_watts):
    ui, device = startup_watts
    ui._laser_status_in_progress = True
    ui._use_native_ir_control = Mock()
    ui._complete_laser_startup()
    assert ui._laser_startup_watts_pending and not device.calls
    ui._laser_startup_retry_timer.start.assert_called_once_with(250)
    ui._use_native_ir_control.assert_not_called()


@pytest.mark.parametrize('action', ['standby', 'aom', 'protocol'])
def test_manual_actions_cancel_queued_startup_handover(startup_watts, action):
    ui, device = startup_watts
    if action == 'standby':
        ui.laser_standby_clicked()
    elif action == 'aom':
        ui._toggle_laser_power_control()
    else:
        with ui._laser_mode_transition('manual protocol') as port:
            assert port == 'FAKE'
    ui._use_native_ir_control = Mock()
    ui._complete_laser_startup()
    assert not ui._laser_startup_watts_pending
    ui._use_native_ir_control.assert_not_called()
    assert not device.calls


def test_startup_rechecks_listen_immediately_before_protocol_write(startup_watts, monkeypatch):
    ui, device = startup_watts
    device.StatusRead = Mock(side_effect=['ly_oxp2_dev_status 9', 'ly_oxp2_dev_status 33'])
    bus = Mock()
    monkeypatch.setattr(nktpbus_switch, 'switch_to_nktpbus', bus)
    ui._complete_laser_startup()
    bus.assert_not_called()
    device.close_port.assert_not_called()
    assert device.code == 9 and not device.calls
    assert 'requires confirmed Listen' in ui.errors[-1]
    assert not ui._laser_startup_watts_pending
