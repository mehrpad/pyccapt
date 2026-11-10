"""Powered-off laser polling stays invalid without repeating its diagnostic."""
import logging
from unittest.mock import Mock

import pytest

from pyccapt.control.core.control_state import Connection
from test_control_states import make_variables
from test_laser_manual_contract import laser_gui, mode_gui  # noqa: F401


def test_alternating_offline_sdk_results_log_one_episode_across_all_channels(mode_gui, caplog, capsys):
    ui = mode_gui
    ui.variables = make_variables()
    failures = [ConnectionError('Native register 0x01: 6:RegResultTimeout'),
                ConnectionError('Native register 0x67: 13:RegResultPortNotFound')] * 5
    ui.check_laser_status = Mock(side_effect=failures)
    with caplog.at_level(logging.INFO):
        for _ in failures:
            ui._poll_laser_status()
    warnings = [record for record in caplog.records if record.name == 'pyccapt.laser'
                and record.levelno == logging.WARNING]
    assert len(warnings) == 1 and 'powered off or disconnected' in warnings[0].message
    laser_faults = [record for record in caplog.records if record.name == 'pyccapt.state'
                    and record.message.startswith('laser owner=') and 'event=fault' in record.message]
    assert len(laser_faults) == 1
    assert 'Laser status poll failed' not in capsys.readouterr().out
    assert ui.variables.control_state('laser').connection == Connection.DISCONNECTED
    assert not ui.variables.control_state('laser').valid
    assert not ui.variables.laser_telemetry['valid']
    assert not ui.laser_on.isEnabled() and not ui.laser_standby.isEnabled()
    assert len(ui.errors) == 1
    assert not ui._laser_status_in_progress


def test_valid_replies_log_one_recovery_and_allow_a_new_offline_episode(mode_gui, caplog):
    ui = mode_gui
    original_check = ui.check_laser_status
    ui.check_laser_status = Mock(side_effect=TimeoutError('No serial reply'))
    with caplog.at_level(logging.INFO, logger='pyccapt.laser'):
        ui._poll_laser_status()
        ui._poll_laser_status()
        ui.check_laser_status = original_check
        ui._poll_laser_status()
        ui._poll_laser_status()
        ui.check_laser_status = Mock(side_effect=ConnectionError('Native register 0x01: 13:RegResultPortNotFound'))
        ui._poll_laser_status()
        ui._poll_laser_status()
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 2
    recoveries = [record for record in caplog.records if record.message.startswith('Laser communication restored')]
    assert len(recoveries) == 1
    assert ui._last_laser_poll_error.startswith('Laser unavailable on ')


def test_distinct_non_transport_faults_are_still_reported(mode_gui, caplog):
    ui = mode_gui
    ui.check_laser_status = Mock(side_effect=[ValueError('Bad status'), ValueError('Bad status'),
                                            ConnectionError('Native register 0x01: 15:RegResultApplicationBusy')])
    with caplog.at_level(logging.WARNING, logger='pyccapt.laser'):
        for _ in range(3):
            ui._poll_laser_status()
    assert len(caplog.records) == 2
    assert 'Bad status' in caplog.records[0].message
    assert 'RegResultApplicationBusy' in caplog.records[1].message
    assert len(ui.errors) == 2


def test_no_recovery_is_claimed_when_there_is_no_connected_device(mode_gui, caplog):
    ui = mode_gui
    ui.laser_device = None
    ui._last_laser_poll_error = 'Laser unavailable on FAKE; it may be powered off or disconnected.'
    with caplog.at_level(logging.INFO, logger='pyccapt.laser'):
        ui._poll_laser_status()
    assert not any('communication restored' in record.message for record in caplog.records)
    assert ui._last_laser_poll_error
