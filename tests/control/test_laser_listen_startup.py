"""Startup/shutdown select Listen; LEDs still require actual status replies."""
from unittest.mock import Mock

import pytest
from PyQt6 import QtWidgets

from pyccapt.control.nkt_photonics import origamiClassCLI, readback
from test_laser_manual_contract import Laser, laser_gui, mode_gui  # noqa: F401


@pytest.mark.parametrize('initial', [1, 9, 17, 33, 65, 129])
def test_initial_cli_connection_requests_only_listen_when_needed(mode_gui, monkeypatch, initial):
    ui = mode_gui
    device = Laser()
    device.code = initial
    device.open_port = Mock(return_value=0)
    monkeypatch.setattr(origamiClassCLI, 'origClass', lambda port: device)
    assert ui._open_laser_cli('FAKE', initial_open=True)
    assert device.calls == ([] if initial == 9 else [('Listen',)])
    assert readback.scalar(ui._last_laser_status) == 9
    assert ui.led_laser_listen.pixmap().cacheKey() == ui.led_green.cacheKey()
    assert ui.led_laser_laser_standby.pixmap().cacheKey() == ui.led_red.cacheKey()
    assert not ui.laser_on.isEnabled()


@pytest.mark.parametrize('still_observed', [17, 33])
def test_startup_listen_ack_does_not_invent_confirmation(mode_gui, monkeypatch, still_observed):
    ui = mode_gui
    ui.laser_state_label = QtWidgets.QLabel()
    device = Laser()
    device.code = still_observed
    device.Listen = Mock()  # Accepted command, transition has not completed.
    device.open_port = Mock(return_value=0)
    monkeypatch.setattr(origamiClassCLI, 'origClass', lambda port: device)
    assert ui._open_laser_cli('FAKE', initial_open=True)
    assert readback.scalar(ui._last_laser_status) == still_observed
    assert ui.laser_state_label.text().startswith('Laser: Listen requested / ')
    assert not ui.laser_on.isEnabled() and not ui.laser_standby.isEnabled()
    before = ui.led_laser_listen.pixmap().cacheKey()
    ui._blink_laser_indicators()
    assert {before, ui.led_laser_listen.pixmap().cacheKey()} == {
        ui.led_red.cacheKey(), ui.led_orange.cacheKey()}
    device.code = 9
    ui.check_laser_status()
    device.Listen.assert_called_once()
    assert ui.led_laser_listen.pixmap().cacheKey() == ui.led_green.cacheKey()
    assert ui.laser_standby.isEnabled()
    assert 'Listen requested' not in ui.laser_state_label.text()


def test_failed_startup_listen_is_reported_instead_of_showing_listen(mode_gui, monkeypatch):
    ui = mode_gui
    device = Laser()
    device.open_port = Mock(return_value=0)
    device.Listen = Mock(side_effect=TimeoutError('Listen request timed out'))
    monkeypatch.setattr(origamiClassCLI, 'origClass', lambda port: device)
    assert not ui._open_laser_cli('FAKE', initial_open=True)
    assert ui.laser_device is None
    assert 'Listen request timed out' in ui.laser_connection_banner.text()
    assert ui.led_laser_listen.pixmap().cacheKey() != ui.led_green.cacheKey()
    assert not device.calls


def test_shutdown_leaves_listen_instead_of_requesting_standby(mode_gui):
    ui = mode_gui
    device = ui.laser_device
    device.code = 129
    ui._cancel_laser_alignment = Mock()
    ui.stop()
    assert device.calls == [('AOMDisable',), ('AOM', 0), ('Listen',)]
    assert device.code == 9
    device.close_port.assert_called_once()
    assert ui.laser_device is None
